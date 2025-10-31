#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>



namespace {

constexpr double kMinProb = 1e-12;

inline double safe_log(double p) {
    if (p <= 0.0) {
        return std::log(kMinProb);
    }
    return std::log(p);
}

struct CLIOptions {
    std::string bif_input;
    std::string data_input;
    std::string bif_output;
    int iters = 10;
    bool check = false; // keep an internal CPT sanity check toggle (not exposed)
};

class ArgParser {
  public:
    ArgParser(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            args_.emplace_back(argv[i]);
        }
    }

    CLIOptions parse() {
        CLIOptions opt;
        // Only accept three positional arguments: input.bif, data.dat, output.bif
        if (args_.size() != 3) {
            throw std::runtime_error("Usage: learn_bn <input.bif> <data.dat> <output.bif>");
        }
        opt.bif_input = args_[0];
        opt.data_input = args_[1];
        opt.bif_output = args_[2];
        return opt;
    }

  private:
    std::vector<std::string> args_;
};

struct Token {
    enum class Type { Identifier, Symbol, End };
    Type type = Type::End;
    std::string text;
    char symbol = '\0';
};

class Tokenizer {
  public:
    explicit Tokenizer(const std::string& input) : data_(input), pos_(0) {}

    Token next() {
        if (has_peek_) {
            has_peek_ = false;
            return peek_token_;
        }
        return readToken();
    }

    Token peek() {
        if (!has_peek_) {
            peek_token_ = readToken();
            has_peek_ = true;
        }
        return peek_token_;
    }

    void expectSymbol(char c) {
        Token t = next();
        if (t.type != Token::Type::Symbol || t.symbol != c) {
            std::ostringstream oss;
            oss << "Expected symbol '" << c << "'";
            throw std::runtime_error(oss.str());
        }
    }

    Token expectIdentifier() {
        Token t = next();
        if (t.type != Token::Type::Identifier) {
            throw std::runtime_error("Expected identifier");
        }
        return t;
    }

  private:
    bool isSymbol(char c) const {
        switch (c) {
            case '{':
            case '}':
            case '(':
            case ')':
            case '[':
            case ']':
            case ',':
            case ';':
            case '|':
            case '=':
                return true;
            default:
                return false;
        }
    }

    void skipWhitespace() {
        while (pos_ < data_.size()) {
            char c = data_[pos_];
            if (std::isspace(static_cast<unsigned char>(c))) {
                ++pos_;
                continue;
            }
            if (c == '/' && pos_ + 1 < data_.size()) {
                char d = data_[pos_ + 1];
                if (d == '/') {
                    pos_ += 2;
                    while (pos_ < data_.size() && data_[pos_] != '\n') {
                        ++pos_;
                    }
                    continue;
                }
                if (d == '*') {
                    pos_ += 2;
                    while (pos_ + 1 < data_.size()) {
                        if (data_[pos_] == '*' && data_[pos_ + 1] == '/') {
                            pos_ += 2;
                            break;
                        }
                        ++pos_;
                    }
                    continue;
                }
            }
            break;
        }
    }

    Token readToken() {
        skipWhitespace();
        if (pos_ >= data_.size()) {
            return Token{Token::Type::End, "", '\0'};
        }
        char c = data_[pos_];
        if (isSymbol(c)) {
            ++pos_;
            return Token{Token::Type::Symbol, "", c};
        }
        if (c == '"' || c == '\'') {
            char quote = c;
            ++pos_;
            std::string text;
            while (pos_ < data_.size() && data_[pos_] != quote) {
                text.push_back(data_[pos_++]);
            }
            if (pos_ >= data_.size()) {
                throw std::runtime_error("Unterminated quoted string");
            }
            ++pos_;
            return Token{Token::Type::Identifier, text, '\0'};
        }
        std::string text;
        while (pos_ < data_.size()) {
            char ch = data_[pos_];
            if (std::isspace(static_cast<unsigned char>(ch)) || isSymbol(ch)) {
                break;
            }
            text.push_back(ch);
            ++pos_;
        }
        return Token{Token::Type::Identifier, text, '\0'};
    }

    std::string data_;
    size_t pos_;
    bool has_peek_ = false;
    Token peek_token_;
};

struct Variable {
    std::string name;
    std::vector<std::string> states;
    std::vector<std::string> parent_names;  // temporary before resolving indices
    std::unordered_map<std::string, int> state_to_index;
    std::vector<int> parents;
    std::vector<int> children;
    std::vector<size_t> parent_strides;
    size_t cpt_offset = 0;
    size_t q = 1;  // number of parent configurations
    int r = 0;     // number of states
    std::vector<double> raw_cpt_values;
};

struct BN {
    std::vector<Variable> vars;
    std::unordered_map<std::string, int> name_to_index;
    size_t total_cpt_entries = 0;
    size_t max_states = 0;
    std::vector<double> fixed_values;
    std::vector<uint8_t> fixed_mask;
};

struct Data {
    std::vector<int32_t> values;
    size_t nrows = 0;
    size_t nvars = 0;
    size_t complete_rows = 0;
    size_t single_missing_rows = 0;

    const int32_t* row(size_t idx) const {
        return values.data() + idx * nvars;
    }
};

std::string readFileToString(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Variable parseVariable(Tokenizer& tz) {
    Token name_tok = tz.expectIdentifier();
    Variable var;
    var.name = name_tok.text;
    tz.expectSymbol('{');
    bool states_parsed = false;
    while (true) {
        Token t = tz.next();
        if (t.type == Token::Type::End) {
            throw std::runtime_error("Unexpected EOF in variable block");
        }
        if (t.type == Token::Type::Symbol && t.symbol == '}') {
            break;
        }
        if (t.type == Token::Type::Identifier && t.text == "type") {
            Token discrete = tz.expectIdentifier();
            if (discrete.text != "discrete") {
                throw std::runtime_error("Only discrete variables are supported");
            }
            tz.expectSymbol('[');
            Token num_tok = tz.expectIdentifier();
            int nstates = std::stoi(num_tok.text);
            tz.expectSymbol(']');
            Token maybe_eq = tz.next();
            if (!(maybe_eq.type == Token::Type::Symbol && (maybe_eq.symbol == '=' || maybe_eq.symbol == '{'))) {
                throw std::runtime_error("Expected '=' or '{' after discrete declaration");
            }
            if (maybe_eq.symbol != '{') {
                tz.expectSymbol('{');
            }
            var.states.reserve(static_cast<size_t>(nstates));
            for (int i = 0; i < nstates; ++i) {
                Token state_tok = tz.expectIdentifier();
                var.states.push_back(state_tok.text);
                if (i + 1 < nstates) {
                    tz.expectSymbol(',');
                }
            }
            tz.expectSymbol('}');
            Token term = tz.next();
            if (!(term.type == Token::Type::Symbol && term.symbol == ';')) {
                throw std::runtime_error("Expected ';' after state list");
            }
            states_parsed = true;
        }
    }
    if (!states_parsed) {
        throw std::runtime_error("Variable block missing states");
    }
    return var;
}

void skipBlock(Tokenizer& tz) {
    tz.expectSymbol('{');
    int depth = 1;
    while (depth > 0) {
        Token t = tz.next();
        if (t.type == Token::Type::End) {
            throw std::runtime_error("Unexpected EOF while skipping block");
        }
        if (t.type == Token::Type::Symbol) {
            if (t.symbol == '{') {
                ++depth;
            } else if (t.symbol == '}') {
                --depth;
            }
        }
    }
    tz.expectSymbol(';');
}

bool is_number_token(const std::string& text) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0';
}

double parse_probability_value(const std::string& text) {
    char* end = nullptr;
    double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid probability value '" + text + "'");
    }
    return value;
}

void parse_probability_value_list(Tokenizer& tz, std::vector<double>& dest, const Token* first_token) {
    Token current;
    if (first_token) {
        current = *first_token;
    } else {
        current = tz.expectIdentifier();
    }
    while (true) {
        if (!is_number_token(current.text)) {
            throw std::runtime_error("Expected numeric probability value");
        }
        dest.push_back(parse_probability_value(current.text));
        Token sep = tz.next();
        if (sep.type == Token::Type::Symbol && sep.symbol == ',') {
            current = tz.expectIdentifier();
            continue;
        }
        if (sep.type == Token::Type::Symbol && sep.symbol == ';') {
            break;
        }
        throw std::runtime_error("Expected ',' or ';' after probability value");
    }
}

void parse_probability_body(Tokenizer& tz, Variable& var) {
    tz.expectSymbol('{');
    var.raw_cpt_values.clear();
    while (true) {
        Token t = tz.next();
        if (t.type == Token::Type::End) {
            throw std::runtime_error("Unexpected EOF inside probability block for variable " + var.name);
        }
        if (t.type == Token::Type::Symbol) {
            if (t.symbol == '}') {
                break;
            }
            if (t.symbol == '(') {
                int depth = 1;
                while (depth > 0) {
                    Token inner = tz.next();
                    if (inner.type == Token::Type::End) {
                        throw std::runtime_error("Unexpected EOF inside parent assignment for " + var.name);
                    }
                    if (inner.type == Token::Type::Symbol) {
                        if (inner.symbol == '(') {
                            ++depth;
                        } else if (inner.symbol == ')') {
                            --depth;
                        }
                    }
                }
                parse_probability_value_list(tz, var.raw_cpt_values, nullptr);
                continue;
            }
            if (t.symbol == ';') {
                continue;
            }
        }
        if (t.type == Token::Type::Identifier) {
            if (t.text == "table" || t.text == "default") {
                parse_probability_value_list(tz, var.raw_cpt_values, nullptr);
                continue;
            }
            if (is_number_token(t.text)) {
                parse_probability_value_list(tz, var.raw_cpt_values, &t);
                continue;
            }
        }
        throw std::runtime_error("Unexpected token '" + t.text + "' in probability block for variable " + var.name);
    }
    tz.expectSymbol(';');
}

void parseProbability(Tokenizer& tz, BN& bn) {
    tz.expectSymbol('(');
    Token var_tok = tz.expectIdentifier();
    auto it = bn.name_to_index.find(var_tok.text);
    if (it == bn.name_to_index.end()) {
        throw std::runtime_error("Probability block references unknown variable " + var_tok.text);
    }
    int var_idx = it->second;
    Variable& var = bn.vars[var_idx];
    Token tok = tz.next();
    if (tok.type == Token::Type::Symbol && tok.symbol == '|') {
        while (true) {
            Token parent_tok = tz.expectIdentifier();
            var.parent_names.push_back(parent_tok.text);
            Token next_tok = tz.next();
            if (next_tok.type == Token::Type::Symbol && next_tok.symbol == ',') {
                continue;
            }
            if (next_tok.type == Token::Type::Symbol && next_tok.symbol == ')') {
                tok = next_tok;
                break;
            }
            throw std::runtime_error("Malformed parent list in probability block");
        }
    } else if (tok.type == Token::Type::Symbol && tok.symbol == ')') {
        // no parents
    } else {
        throw std::runtime_error("Malformed probability header");
    }
    if (!(tok.type == Token::Type::Symbol && tok.symbol == ')')) {
        tz.expectSymbol(')');
    }
    parse_probability_body(tz, var);
}

BN parse_bif(const std::string& path) {
    std::string content = readFileToString(path);
    Tokenizer tz(content);
    BN bn;
    while (true) {
        Token t = tz.next();
        if (t.type == Token::Type::End) {
            break;
        }
        if (t.type == Token::Type::Identifier) {
            if (t.text == "variable") {
                Variable var = parseVariable(tz);
                var.r = static_cast<int>(var.states.size());
                var.state_to_index.reserve(var.states.size());
                for (size_t i = 0; i < var.states.size(); ++i) {
                    var.state_to_index[var.states[i]] = static_cast<int>(i);
                }
                int idx = static_cast<int>(bn.vars.size());
                bn.name_to_index[var.name] = idx;
                bn.vars.emplace_back(std::move(var));
            } else if (t.text == "probability") {
                if (bn.vars.empty()) {
                    throw std::runtime_error("Encountered probability block before variables");
                }
                parseProbability(tz, bn);
            } else if (t.text == "network") {
                skipBlock(tz);
            } else {
                throw std::runtime_error("Unexpected token in BIF: " + t.text);
            }
        } else {
            // Ignore stray symbols/separators at top-level.
        }
    }

    // Resolve parents and compute strides.
    bn.total_cpt_entries = 0;
    bn.max_states = 0;
    for (size_t vidx = 0; vidx < bn.vars.size(); ++vidx) {
        Variable& var = bn.vars[vidx];
        bn.max_states = std::max(bn.max_states, static_cast<size_t>(var.r));
        var.parents.clear();
        var.q = 1;
        var.parent_strides.clear();
        if (!var.parent_names.empty()) {
            var.parents.reserve(var.parent_names.size());
            for (const std::string& pname : var.parent_names) {
                auto pit = bn.name_to_index.find(pname);
                if (pit == bn.name_to_index.end()) {
                    throw std::runtime_error("Unknown parent " + pname + " for variable " + var.name);
                }
                var.parents.push_back(pit->second);
            }
            var.parent_strides.resize(var.parents.size());
            size_t stride = 1;
            for (size_t j = var.parents.size(); j-- > 0;) {
                var.parent_strides[j] = stride;
                stride *= static_cast<size_t>(bn.vars[static_cast<size_t>(var.parents[j])].r);
            }
            var.q = stride;
        }
        bn.total_cpt_entries += static_cast<size_t>(var.r) * var.q;
        var.parent_names.clear();
    }
    // Build children lists.
    for (size_t vidx = 0; vidx < bn.vars.size(); ++vidx) {
        Variable& var = bn.vars[vidx];
        for (int parent : var.parents) {
            bn.vars[static_cast<size_t>(parent)].children.push_back(static_cast<int>(vidx));
        }
    }
    // Assign CPT offsets.
    size_t offset = 0;
    for (auto& var : bn.vars) {
        var.cpt_offset = offset;
        offset += static_cast<size_t>(var.r) * var.q;
    }
    bn.fixed_mask.assign(bn.total_cpt_entries, 0);
    bn.fixed_values.assign(bn.total_cpt_entries, 0.0);
    for (auto& var : bn.vars) {
        size_t expected = static_cast<size_t>(var.r) * var.q;
        if (!var.raw_cpt_values.empty()) {
            if (var.raw_cpt_values.size() != expected) {
                std::ostringstream oss;
                oss << "Probability table for variable " << var.name
                    << " has " << var.raw_cpt_values.size() << " entries but expected "
                    << expected;
                throw std::runtime_error(oss.str());
            }
            for (size_t i = 0; i < expected; ++i) {
                double value = var.raw_cpt_values[i];
                if (value >= 0.0) {
                    bn.fixed_mask[var.cpt_offset + i] = 1;
                    bn.fixed_values[var.cpt_offset + i] = value;
                }
            }
        }
        var.raw_cpt_values.clear();
        var.raw_cpt_values.shrink_to_fit();
    }
    return bn;
}

char detect_delimiter(const std::string& line) {
    const std::vector<char> candidates = {',', '\t', ' '};
    size_t best_count = 0;
    char best = ',';
    for (char cand : candidates) {
        size_t count = 0;
        bool in_quotes = false;
        for (char c : line) {
            if (c == '"' || c == '\'') {
                in_quotes = !in_quotes;
            } else if (!in_quotes && c == cand) {
                ++count;
            }
        }
        if (count > best_count) {
            best_count = count;
            best = cand;
        }
    }
    return best;
}

void split_line(const std::string& line, char delim, std::vector<std::string>& tokens) {
    tokens.clear();
    std::string current;
    bool in_quotes = false;
    char quote_char = '\0';
    auto flush = [&]() {
        tokens.push_back(current);
        current.clear();
    };
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if ((c == '"' || c == '\'') && !in_quotes) {
            in_quotes = true;
            quote_char = c;
            continue;
        }
        if (in_quotes && c == quote_char) {
            in_quotes = false;
            quote_char = '\0';
            continue;
        }
        if (!in_quotes) {
            if (c == delim || (delim == ' ' && (c == ' ' || c == '\t'))) {
                flush();
                if (delim == ' ') {
                    while (i + 1 < line.size() && (line[i + 1] == ' ' || line[i + 1] == '\t')) {
                        ++i;
                    }
                }
                continue;
            }
        }
        current.push_back(c);
    }
    flush();
}

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

bool is_integer_token(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    for (char c : token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

Data load_data(const std::string& path, const BN& bn) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        lines.push_back(line);
    }
    if (lines.empty()) {
        throw std::runtime_error("No data rows found in " + path);
    }
    char delim = detect_delimiter(lines.front());
    Data data;
    data.nvars = bn.vars.size();
    data.nrows = lines.size();
    data.values.resize(data.nrows * data.nvars, -1);
    std::vector<std::string> tokens;
    tokens.reserve(data.nvars);
    for (size_t row = 0; row < lines.size(); ++row) {
        split_line(lines[row], delim, tokens);
        if (tokens.size() != data.nvars) {
            std::ostringstream oss;
            oss << "Row " << row << " has " << tokens.size()
                << " tokens but expected " << data.nvars;
            throw std::runtime_error(oss.str());
        }
        int missing = 0;
        for (size_t col = 0; col < data.nvars; ++col) {
            std::string token = trim(tokens[col]);
            if (!token.empty() && (token.front() == '"' || token.front() == '\'')) {
                token.erase(token.begin());
            }
            if (!token.empty() && (token.back() == '"' || token.back() == '\'')) {
                token.pop_back();
            }
            if (token.empty() || token == "?" || token == "-1") {
                data.values[row * data.nvars + col] = -1;
                ++missing;
                continue;
            }
            int32_t value = -1;
            if (is_integer_token(token)) {
                value = static_cast<int32_t>(std::stoi(token));
            } else {
                const auto& var = bn.vars[col];
                auto it = var.state_to_index.find(token);
                if (it == var.state_to_index.end()) {
                    std::ostringstream oss;
                    oss << "Unknown state '" << token << "' for variable " << var.name
                        << " at row " << row;
                    throw std::runtime_error(oss.str());
                }
                value = static_cast<int32_t>(it->second);
            }
            if (value < 0 || value >= bn.vars[col].r) {
                std::ostringstream oss;
                oss << "State index out of range for variable " << bn.vars[col].name
                    << " at row " << row;
                throw std::runtime_error(oss.str());
            }
            data.values[row * data.nvars + col] = value;
        }
        if (missing == 0) {
            ++data.complete_rows;
        }
        if (missing == 1) {
            ++data.single_missing_rows;
        }
    }
    return data;
}

inline size_t family_index(const Variable& var, const int32_t* row,
                           const std::vector<Variable>& vars, int override_var,
                           int override_value, bool& ok) {
    ok = true;
    size_t idx = 0;
    for (size_t i = 0; i < var.parents.size(); ++i) {
        int parent = var.parents[i];
        int32_t val = (parent == override_var) ? override_value : row[parent];
        if (val < 0) {
            ok = false;
            return 0;
        }
        idx += var.parent_strides[i] * static_cast<size_t>(val);
    }
    return idx;
}

inline bool accumulate_family(const std::vector<Variable>& vars, int var_index,
                              const int32_t* row, int override_var, int override_value,
                              double weight, std::vector<double>& counts) {
    const Variable& var = vars[static_cast<size_t>(var_index)];
    int32_t val = (var_index == override_var) ? override_value : row[var_index];
    if (val < 0) {
        return false;
    }
    bool ok = true;
    size_t parent_idx = family_index(var, row, vars, override_var, override_value, ok);
    if (!ok) {
        return false;
    }
    size_t base = var.cpt_offset + parent_idx * static_cast<size_t>(var.r);
    counts[base + static_cast<size_t>(val)] += weight;
    return true;
}

struct RowInfo {
    int first_missing = -1;
    int missing_count = 0;
};

RowInfo analyze_row(const int32_t* row, size_t nvars) {
    RowInfo info;
    for (size_t i = 0; i < nvars; ++i) {
        if (row[i] < 0) {
            if (info.first_missing < 0) {
                info.first_missing = static_cast<int>(i);
            }
            ++info.missing_count;
        }
    }
    return info;
}

void accumulate_complete_row(const BN& bn, const int32_t* row, std::vector<double>& counts,
                             double weight, int override_var, int override_value) {
    for (size_t vidx = 0; vidx < bn.vars.size(); ++vidx) {
        if (static_cast<int>(vidx) != override_var && row[vidx] < 0) {
            continue;
        }
        accumulate_family(bn.vars, static_cast<int>(vidx), row, override_var, override_value,
                          weight, counts);
    }
}

void apply_fixed_entries(const BN& bn, std::vector<double>& theta) {
    if (bn.fixed_mask.empty()) {
        return;
    }
    size_t limit = std::min(bn.fixed_mask.size(), theta.size());
    for (size_t i = 0; i < limit; ++i) {
        if (bn.fixed_mask[i]) {
            theta[i] = bn.fixed_values[i];
        }
    }
}

// Build Markov blanket node lists for each variable (parents, children, co-parents of children, children themselves)
std::vector<std::vector<int>> build_markov_blankets(const BN& bn) {
    const size_t total_variables = bn.vars.size();
    std::vector<std::vector<int>> markov_blanket_map(total_variables);
    
    for (size_t var_index = 0; var_index < total_variables; var_index++) {
        const Variable& target_variable = bn.vars[var_index];
        std::vector<int> blanket_nodes;
        
        // Pre-allocate space for efficiency
        size_t estimated_size = target_variable.parents.size() + (target_variable.children.size() * 3);
        blanket_nodes.reserve(estimated_size);
        
        // Add all parent nodes to the blanket
        for (const auto& parent_node : target_variable.parents) {
            blanket_nodes.push_back(parent_node);
        }
        
        // Add children and their other parents (co-parents)
        for (const auto& child_node : target_variable.children) {
            blanket_nodes.push_back(child_node);
            
            const Variable& child_variable = bn.vars[static_cast<size_t>(child_node)];
            for (const auto& coparent_node : child_variable.parents) {
                if (coparent_node != static_cast<int>(var_index)) {
                    blanket_nodes.push_back(coparent_node);
                }
            }
        }
        
        // Remove duplicates by sorting and erasing unique elements
        std::sort(blanket_nodes.begin(), blanket_nodes.end());
        auto last_unique = std::unique(blanket_nodes.begin(), blanket_nodes.end());
        blanket_nodes.erase(last_unique, blanket_nodes.end());
        
        markov_blanket_map[var_index] = std::move(blanket_nodes);
    }
    
    return markov_blanket_map;
}

// Compute topological order (Kahn's algorithm) using parent edges
std::vector<int> topological_order(const BN& bn) {
    const size_t total_vars = bn.vars.size();
    std::vector<int> incoming_edges(total_vars, 0);
    
    // Calculate in-degree for each node
    for (size_t var_idx = 0; var_idx < total_vars; var_idx++) {
        const auto& current_var = bn.vars[var_idx];
        for (const auto& child_node : current_var.children) {
            incoming_edges[static_cast<size_t>(child_node)]++;
        }
    }
    
    // Initialize queue with nodes having zero in-degree
    std::vector<int> processing_queue;
    processing_queue.reserve(total_vars);
    for (size_t idx = 0; idx < total_vars; idx++) {
        if (incoming_edges[idx] == 0) {
            processing_queue.push_back(static_cast<int>(idx));
        }
    }
    
    // Process nodes in topological order
    std::vector<int> sorted_order;
    sorted_order.reserve(total_vars);
    size_t queue_pos = 0;
    while (queue_pos < processing_queue.size()) {
        int current_node = processing_queue[queue_pos++];
        sorted_order.push_back(current_node);
        
        const auto& node_children = bn.vars[static_cast<size_t>(current_node)].children;
        for (const auto& child : node_children) {
            size_t child_idx = static_cast<size_t>(child);
            incoming_edges[child_idx]--;
            if (incoming_edges[child_idx] == 0) {
                processing_queue.push_back(child);
            }
        }
    }
    
    // Fallback: if cycle detected, return sequential ordering
    if (sorted_order.size() != total_vars) {
        sorted_order.clear();
        for (size_t idx = 0; idx < total_vars; idx++) {
            sorted_order.push_back(static_cast<int>(idx));
        }
    }
    
    return sorted_order;
}

struct Buckets {
    std::vector<size_t> complete;
    std::vector<std::vector<size_t>> per_var_sorted;
};

Buckets build_buckets(const BN& bn, const Data& data, const std::vector<std::vector<int>>& mb) {
    Buckets bucket_structure;
    const size_t num_variables = bn.vars.size();
    bucket_structure.per_var_sorted.assign(num_variables, {});
    
    // Phase 1: Collect unsorted rows per variable
    std::vector<std::vector<size_t>> unsorted_per_variable(num_variables);
    
    for (size_t current_row = 0; current_row < data.nrows; current_row++) {
        const int32_t* row_data = data.row(current_row);
        RowInfo row_info = analyze_row(row_data, data.nvars);
        
        if (row_info.missing_count == 0) {
            bucket_structure.complete.push_back(current_row);
        } else {
            if (row_info.first_missing >= 0) {
                size_t missing_var_idx = static_cast<size_t>(row_info.first_missing);
                unsorted_per_variable[missing_var_idx].push_back(current_row);
            }
        }
    }
    
    // Phase 2: Sort each bucket by Markov blanket observational richness
    for (size_t variable_idx = 0; variable_idx < num_variables; variable_idx++) {
        const auto& row_indices = unsorted_per_variable[variable_idx];
        
        if (row_indices.empty()) {
            continue;
        }
        
        // Create richness score pairs: (score, row_index_position)
        std::vector<std::pair<int, int>> richness_pairs;
        richness_pairs.reserve(row_indices.size());
        
        const auto& blanket_nodes = mb[variable_idx];
        
        for (size_t pos = 0; pos < row_indices.size(); pos++) {
            const int32_t* row_data = data.row(row_indices[pos]);
            int richness_score = 0;
            
            // Count observed nodes in Markov blanket
            for (const auto& blanket_node : blanket_nodes) {
                if (row_data[blanket_node] >= 0) {
                    richness_score++;
                }
            }
            
            richness_pairs.emplace_back(richness_score, static_cast<int>(pos));
        }
        
        // Sort by richness in descending order
        std::sort(richness_pairs.begin(), richness_pairs.end(), 
                  [](const auto& first, const auto& second) {
                      return first.first > second.first;
                  });
        
        // Build sorted row indices based on richness ranking
        bucket_structure.per_var_sorted[variable_idx].reserve(row_indices.size());
        for (const auto& pair : richness_pairs) {
            size_t original_pos = static_cast<size_t>(pair.second);
            bucket_structure.per_var_sorted[variable_idx].push_back(row_indices[original_pos]);
        }
    }
    
    return bucket_structure;
}


// Soft E-step bucketed
void soft_e_step_bucketed(const BN& bn, const Data& data, const std::vector<double>& theta,
                          const Buckets& buckets, const std::vector<int>& visit_order,
                          std::vector<double>& counts, int /*threads*/) {
    // Initialize counts to zero
    std::fill(counts.begin(), counts.end(), 0.0);
    
    // Allocate workspace for probability computations
    std::vector<double> state_log_scores(bn.max_states, 0.0);
    
    // Phase 1: Process complete rows without missing values
    for (size_t row_iter = 0; row_iter < buckets.complete.size(); row_iter++) {
        const int32_t* complete_row = data.row(buckets.complete[row_iter]);
        accumulate_complete_row(bn, complete_row, counts, 1.0, -1, -1);
    }
    
    // Phase 2: Process bucketed rows with missing values
    for (const auto& target_var : visit_order) {
        const auto& bucket_rows = buckets.per_var_sorted[static_cast<size_t>(target_var)];
        
        if (bucket_rows.empty()) {
            continue;
        }
        
        for (size_t bucket_iter = 0; bucket_iter < bucket_rows.size(); bucket_iter++) {
            const int32_t* current_row = data.row(bucket_rows[bucket_iter]);
            const Variable& missing_variable = bn.vars[static_cast<size_t>(target_var)];
            
            // Check if parent configuration is valid
            bool parents_observed = true;
            size_t parent_configuration = family_index(missing_variable, current_row, bn.vars, 
                                                       -1, -1, parents_observed);
            
            if (!parents_observed) {
                accumulate_complete_row(bn, current_row, counts, 1.0, -1, -1);
                continue;
            }
            
            // Compute log scores for each possible state
            double maximum_log_score = -std::numeric_limits<double>::infinity();
            
            for (int candidate_state = 0; candidate_state < missing_variable.r; candidate_state++) {
                // Start with prior probability for this state
                size_t cpt_index = missing_variable.cpt_offset + parent_configuration * missing_variable.r + candidate_state;
                double score_accumulator = safe_log(theta[cpt_index]);
                
                // Add likelihood from children
                for (const auto& child_var_idx : missing_variable.children) {
                    const Variable& child_variable = bn.vars[static_cast<size_t>(child_var_idx)];
                    int32_t observed_child_state = current_row[child_var_idx];
                    
                    if (observed_child_state < 0) {
                        continue;
                    }
                    
                    // Compute child's parent configuration
                    bool child_parents_valid = true;
                    size_t child_parent_config = 0;
                    
                    for (size_t parent_pos = 0; parent_pos < child_variable.parents.size(); parent_pos++) {
                        int parent_var = child_variable.parents[parent_pos];
                        int32_t parent_value = current_row[parent_var];
                        
                        // Use candidate state if this parent is the missing variable
                        if (parent_var == target_var) {
                            parent_value = candidate_state;
                        }
                        
                        if (parent_value < 0) {
                            child_parents_valid = false;
                            break;
                        }
                        
                        child_parent_config += child_variable.parent_strides[parent_pos] * static_cast<size_t>(parent_value);
                    }
                    
                    if (!child_parents_valid) {
                        continue;
                    }
                    
                    // Look up child probability
                    size_t child_cpt_idx = child_variable.cpt_offset + 
                                          child_parent_config * static_cast<size_t>(child_variable.r) + 
                                          static_cast<size_t>(observed_child_state);
                    double child_probability = theta[child_cpt_idx];
                    score_accumulator += safe_log(child_probability);
                }
                
                state_log_scores[static_cast<size_t>(candidate_state)] = score_accumulator;
                
                if (score_accumulator > maximum_log_score) {
                    maximum_log_score = score_accumulator;
                }
            }
            
            // Convert log scores to probabilities using log-sum-exp trick
            double probability_sum = 0.0;
            
            for (int candidate_state = 0; candidate_state < missing_variable.r; candidate_state++) {
                double normalized_score = std::exp(state_log_scores[static_cast<size_t>(candidate_state)] - maximum_log_score);
                state_log_scores[static_cast<size_t>(candidate_state)] = normalized_score;
                probability_sum += normalized_score;
            }
            
            // Handle degenerate case
            if (probability_sum <= 0.0) {
                accumulate_complete_row(bn, current_row, counts, 1.0, -1, -1);
                continue;
            }
            
            // Accumulate weighted counts for each state
            for (int candidate_state = 0; candidate_state < missing_variable.r; candidate_state++) {
                double state_weight = state_log_scores[static_cast<size_t>(candidate_state)] / probability_sum;
                accumulate_complete_row(bn, current_row, counts, state_weight, target_var, candidate_state);
            }
        }
    }
}


void soft_e_step(const BN& bn, const Data& data, const std::vector<double>& theta,
                 std::vector<double>& counts, int /*threads*/) {
    std::fill(counts.begin(), counts.end(), 0.0);
    std::vector<double> scratch(bn.max_states, 0.0);
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            accumulate_complete_row(bn, row, counts, 1.0, -1, -1);
            continue;
        }
        if (info.first_missing < 0) {
            accumulate_complete_row(bn, row, counts, 1.0, -1, -1);
            continue;
        }
        const Variable& target = bn.vars[static_cast<size_t>(info.first_missing)];
        bool parent_ok = true;
        size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
        if (!parent_ok) {
            accumulate_complete_row(bn, row, counts, 1.0, -1, -1);
            continue;
        }
        double max_log = -std::numeric_limits<double>::infinity();
        for (int x = 0; x < target.r; ++x) {
            double log_score = safe_log(theta[target.cpt_offset + parent_idx * target.r + x]);
            for (int child_idx : target.children) {
                const Variable& child = bn.vars[static_cast<size_t>(child_idx)];
                int32_t child_val = row[child_idx];
                if (child_val < 0) {
                    continue;
                }
                bool child_ok = true;
                size_t child_parent_idx = 0;
                for (size_t j = 0; j < child.parents.size(); ++j) {
                    int parent = child.parents[j];
                    int32_t val = row[parent];
                    if (parent == info.first_missing) {
                        val = x;
                    }
                    if (val < 0) {
                        child_ok = false;
                        break;
                    }
                    child_parent_idx += child.parent_strides[j] * static_cast<size_t>(val);
                }
                if (!child_ok) {
                    continue;
                }
                double prob =
                    theta[child.cpt_offset + child_parent_idx * static_cast<size_t>(child.r) +
                          static_cast<size_t>(child_val)];
                log_score += safe_log(prob);
            }
            scratch[static_cast<size_t>(x)] = log_score;
            if (log_score > max_log) {
                max_log = log_score;
            }
        }
        double sum = 0.0;
        for (int x = 0; x < target.r; ++x) {
            scratch[static_cast<size_t>(x)] = std::exp(scratch[static_cast<size_t>(x)] - max_log);
            sum += scratch[static_cast<size_t>(x)];
        }
        if (sum <= 0.0) {
            accumulate_complete_row(bn, row, counts, 1.0, -1, -1);
            continue;
        }
        for (int x = 0; x < target.r; ++x) {
            double weight = scratch[static_cast<size_t>(x)] / sum;
            accumulate_complete_row(bn, row, counts, weight, info.first_missing, x);
        }
    }
}

double effective_alpha(const Variable& var, double user_alpha) {
    if (user_alpha < 0.0) {
        return 1.0 / static_cast<double>(var.r);
    }
    if (user_alpha == 0.0) {
        return 0.0;
    }
    return std::max(user_alpha, 1.0 / static_cast<double>(var.r));
}

void m_step(const BN& bn, const std::vector<double>& counts, double user_alpha,
            bool use_bdeu, double user_ess, std::vector<double>& theta) {
    const size_t total_variables = bn.vars.size();
    
    for (size_t var_index = 0; var_index < total_variables; var_index++) {
        const Variable& current_var = bn.vars[var_index];
        
        // Determine prior strength based on scoring method
        double dirichlet_prior = 0.0;
        double bdeu_prior_cell = 0.0;
        
        if (use_bdeu) {
            double equivalent_sample_size = (user_ess > 0.0) ? user_ess : 5.0;
            double total_configs = static_cast<double>(current_var.r) * static_cast<double>(current_var.q);
            bdeu_prior_cell = equivalent_sample_size / total_configs;
        } else {
            dirichlet_prior = effective_alpha(current_var, user_alpha);
        }
        
        // Process each parent configuration
        for (size_t parent_config = 0; parent_config < current_var.q; parent_config++) {
            size_t config_base_idx = current_var.cpt_offset + parent_config * static_cast<size_t>(current_var.r);
            
            double probability_mass_fixed = 0.0;
            size_t learnable_params = 0;
            double learnable_sum = 0.0;
            
            // First pass: handle fixed entries and accumulate learnable statistics
            for (int state_idx = 0; state_idx < current_var.r; state_idx++) {
                size_t entry_index = config_base_idx + static_cast<size_t>(state_idx);
                
                bool is_fixed_entry = (entry_index < bn.fixed_mask.size() && bn.fixed_mask[entry_index]);
                
                if (is_fixed_entry) {
                    double fixed_prob = bn.fixed_values[entry_index];
                    theta[entry_index] = fixed_prob;
                    probability_mass_fixed += fixed_prob;
                } else {
                    double prior_value = use_bdeu ? bdeu_prior_cell : dirichlet_prior;
                    double posterior_numerator = counts[entry_index] + prior_value;
                    learnable_sum += posterior_numerator;
                    learnable_params++;
                }
            }
            
            // Calculate remaining probability mass for learnable parameters
            double available_mass = 1.0 - probability_mass_fixed;
            
            if (available_mass < 0.0) {
                available_mass = 0.0;
            } else if (available_mass > 1.0) {
                available_mass = 1.0;
            }
            
            // Skip if no learnable parameters
            if (learnable_params == 0) {
                continue;
            }
            
            // Second pass: assign probabilities to learnable parameters
            if (learnable_sum <= 0.0) {
                // Fallback to uniform distribution over learnable states
                double uniform_prob = (learnable_params > 0) ? (available_mass / static_cast<double>(learnable_params)) : 0.0;
                
                for (int state_idx = 0; state_idx < current_var.r; state_idx++) {
                    size_t entry_index = config_base_idx + static_cast<size_t>(state_idx);
                    bool is_learnable = (entry_index < bn.fixed_mask.size() && !bn.fixed_mask[entry_index]);
                    
                    if (is_learnable) {
                        theta[entry_index] = uniform_prob;
                    }
                }
            } else {
                // Normalize by posterior counts
                for (int state_idx = 0; state_idx < current_var.r; state_idx++) {
                    size_t entry_index = config_base_idx + static_cast<size_t>(state_idx);
                    
                    bool is_fixed_entry = (entry_index < bn.fixed_mask.size() && bn.fixed_mask[entry_index]);
                    
                    if (is_fixed_entry) {
                        continue;
                    }
                    
                    double prior_value = use_bdeu ? bdeu_prior_cell : dirichlet_prior;
                    double posterior_numerator = counts[entry_index] + prior_value;
                    double normalized_prob = (available_mass <= 0.0) ? 0.0 : (available_mass * posterior_numerator / learnable_sum);
                    theta[entry_index] = normalized_prob;
                }
            }
        }
    }
}

void uniform_initialize(const BN& bn, std::vector<double>& theta) {
    for (const auto& var : bn.vars) {
        double uniform = 1.0 / static_cast<double>(var.r);
        for (size_t u = 0; u < var.q; ++u) {
            size_t base = var.cpt_offset + u * static_cast<size_t>(var.r);
            for (int s = 0; s < var.r; ++s) {
                theta[base + static_cast<size_t>(s)] = uniform;
            }
        }
    }
    apply_fixed_entries(bn, theta);
}

void random_dirichlet_initialize(const BN& bn, std::vector<double>& theta, std::mt19937& rng) {
    std::gamma_distribution<double> gamma(1.0);
    for (const auto& var : bn.vars) {
        for (size_t u = 0; u < var.q; ++u) {
            size_t base = var.cpt_offset + u * static_cast<size_t>(var.r);
            double sum = 0.0;
            for (int s = 0; s < var.r; ++s) {
                double g = gamma(rng) + 1e-9; // ensure strictly positive
                theta[base + static_cast<size_t>(s)] = g;
                sum += g;
            }
            double inv = (sum > 0.0) ? (1.0 / sum) : 1.0 / static_cast<double>(var.r);
            for (int s = 0; s < var.r; ++s) {
                theta[base + static_cast<size_t>(s)] *= inv;
            }
        }
    }
    apply_fixed_entries(bn, theta);
}

void initialize_from_complete_rows(const BN& bn, const Data& data, double user_alpha,
                                   bool use_bdeu, double user_ess, std::vector<double>& theta) {
    std::vector<double> counts(bn.total_cpt_entries, 0.0);
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            accumulate_complete_row(bn, row, counts, 1.0, -1, -1);
        }
    }
    m_step(bn, counts, user_alpha, use_bdeu, user_ess, theta);
    apply_fixed_entries(bn, theta);
}

double compute_row_logprob(const BN& bn, const std::vector<double>& theta,
                           const int32_t* row) {
    double logp = 0.0;
    for (size_t vidx = 0; vidx < bn.vars.size(); ++vidx) {
        const Variable& var = bn.vars[vidx];
        int32_t val = row[vidx];
        if (val < 0) {
            continue;
        }
        bool ok = true;
        size_t parent_idx = family_index(var, row, bn.vars, -1, -1, ok);
        if (!ok) {
            continue;
        }
        double prob =
            theta[var.cpt_offset + parent_idx * static_cast<size_t>(var.r) + static_cast<size_t>(val)];
        logp += safe_log(prob);
    }
    return logp;
}

double compute_loglik(const BN& bn, const Data& data, const std::vector<double>& theta,
                      int /*threads*/) {
    double total_loglik = 0.0;
    std::vector<double> log_scores(bn.max_states, 0.0);
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            total_loglik += compute_row_logprob(bn, theta, row);
            continue;
        }
        if (info.first_missing < 0) {
            total_loglik += compute_row_logprob(bn, theta, row);
            continue;
        }
        const Variable& target = bn.vars[static_cast<size_t>(info.first_missing)];
        bool parent_ok = true;
        size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
        if (!parent_ok) {
            total_loglik += compute_row_logprob(bn, theta, row);
            continue;
        }
        double max_log = -std::numeric_limits<double>::infinity();
        for (int x = 0; x < target.r; ++x) {
            double log_score = safe_log(theta[target.cpt_offset + parent_idx * target.r + x]);
            for (int child_idx : target.children) {
                const Variable& child = bn.vars[static_cast<size_t>(child_idx)];
                int32_t child_val = row[child_idx];
                if (child_val < 0) {
                    continue;
                }
                bool child_ok = true;
                size_t child_parent_idx = 0;
                for (size_t j = 0; j < child.parents.size(); ++j) {
                    int parent = child.parents[j];
                    int32_t val = row[parent];
                    if (parent == info.first_missing) {
                        val = x;
                    }
                    if (val < 0) {
                        child_ok = false;
                        break;
                    }
                    child_parent_idx += child.parent_strides[j] * static_cast<size_t>(val);
                }
                if (!child_ok) {
                    continue;
                }
                double prob =
                    theta[child.cpt_offset + child_parent_idx * static_cast<size_t>(child.r) +
                          static_cast<size_t>(child_val)];
                log_score += safe_log(prob);
            }
            log_scores[static_cast<size_t>(x)] = log_score;
            if (log_score > max_log) {
                max_log = log_score;
            }
        }
        double sum = 0.0;
        for (int x = 0; x < target.r; ++x) {
            sum += std::exp(log_scores[static_cast<size_t>(x)] - max_log);
        }
        total_loglik += max_log + std::log(std::max(sum, kMinProb));
    }
    return total_loglik;
}

bool check_cpts(const BN& bn, const std::vector<double>& theta) {
    constexpr double tol = 1e-9;
    for (const auto& var : bn.vars) {
        for (size_t u = 0; u < var.q; ++u) {
            size_t base = var.cpt_offset + u * static_cast<size_t>(var.r);
            double sum = 0.0;
            for (int s = 0; s < var.r; ++s) {
                sum += theta[base + static_cast<size_t>(s)];
            }
            if (std::fabs(sum - 1.0) > tol) {
                return false;
            }
        }
    }
    return true;
}

void write_bif(const std::string& path, const BN& bn, const std::vector<double>& theta) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file " + path);
    }
    out << std::fixed << std::setprecision(6);
    for (const auto& var : bn.vars) {
        out << "variable " << var.name << " {\n";
        out << "  type discrete [ " << var.r << " ] = { ";
        for (int s = 0; s < var.r; ++s) {
            out << var.states[static_cast<size_t>(s)];
            if (s + 1 < var.r) {
                out << ", ";
            }
        }
        out << " };\n";
        out << "}\n\n";
    }
    for (const auto& var : bn.vars) {
        out << "probability ( " << var.name;
        if (!var.parents.empty()) {
            out << " | ";
            for (size_t i = 0; i < var.parents.size(); ++i) {
                out << bn.vars[static_cast<size_t>(var.parents[i])].name;
                if (i + 1 < var.parents.size()) {
                    out << ", ";
                }
            }
        }
        out << " ) {\n";
        if (var.parents.empty()) {
            out << "    table ";
            for (int s = 0; s < var.r; ++s) {
                out << theta[var.cpt_offset + static_cast<size_t>(s)];
                if (s + 1 < var.r) {
                    out << ", ";
                }
            }
            out << ";\n";
        } else {
            std::vector<int> parent_state(var.parents.size(), 0);
            for (size_t u = 0; u < var.q; ++u) {
                out << "    ( ";
                for (size_t j = 0; j < var.parents.size(); ++j) {
                    const auto& parent_var = bn.vars[static_cast<size_t>(var.parents[j])];
                    out << parent_var.states[static_cast<size_t>(parent_state[j])];
                    if (j + 1 < var.parents.size()) {
                        out << ", ";
                    }
                }
                out << " ) ";
                size_t base = var.cpt_offset + u * static_cast<size_t>(var.r);
                for (int s = 0; s < var.r; ++s) {
                    out << theta[base + static_cast<size_t>(s)];
                    if (s + 1 < var.r) {
                        out << ", ";
                    }
                }
                out << ";\n";
                for (int j = static_cast<int>(parent_state.size()) - 1; j >= 0; --j) {
                    const auto& parent_var = bn.vars[static_cast<size_t>(var.parents[static_cast<size_t>(j)])];
                    parent_state[static_cast<size_t>(j)]++;
                    if (parent_state[static_cast<size_t>(j)] >= parent_var.r) {
                        parent_state[static_cast<size_t>(j)] = 0;
                    } else {
                        break;
                    }
                }
            }
        }
        out << "};\n\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        ArgParser parser(argc, argv);
        CLIOptions options = parser.parse();
        BN bn = parse_bif(options.bif_input);
        Data data = load_data(options.data_input, bn);
        // Precompute scheduling and buckets for soft E-step
        auto topo = topological_order(bn);
        auto mb = build_markov_blankets(bn);
        auto buckets = build_buckets(bn, data, mb);

        // Alpha sweep: 0.1, 0.2, ..., 1.0
        std::vector<double> alpha_sweep;
        for (int i = 1; i <= 10; ++i) alpha_sweep.push_back(static_cast<double>(i) / 10.0);

        std::vector<double> best_theta;
        double best_loglik = -std::numeric_limits<double>::infinity();
        double best_alpha = -1.0;
        int best_iters_run = 0;

        for (double this_alpha : alpha_sweep) {
            std::vector<double> theta(bn.total_cpt_entries, 0.0);
            // Initialize from complete rows; falls back to uniform where needed
            initialize_from_complete_rows(bn, data, this_alpha, /*use_bdeu=*/false, /*ess=*/0.0, theta);

            std::vector<double> counts(bn.total_cpt_entries, 0.0);
            double prev_loglik = -std::numeric_limits<double>::infinity();
            double loglik = prev_loglik;
            int iterations_run = 0;
            for (int iter = 0; iter < options.iters; ++iter) {
                soft_e_step_bucketed(bn, data, theta, buckets, topo, counts, /*threads=*/0);
                m_step(bn, counts, this_alpha, /*use_bdeu=*/false, /*ess=*/0.0, theta);
                loglik = compute_loglik(bn, data, theta, /*threads=*/0);
                ++iterations_run;
                if (iter > 0) {
                    double denom = std::max(1.0, std::fabs(prev_loglik));
                    if ((loglik - prev_loglik) / denom < 1e-5) {
                        break;
                    }
                }
                prev_loglik = loglik;
            }
            if (loglik > best_loglik) {
                best_loglik = loglik;
                best_theta = std::move(theta);
                best_alpha = this_alpha;
                best_iters_run = iterations_run;
            }
        }

        if (options.check && !check_cpts(bn, best_theta)) {
            throw std::runtime_error("CPT check failed: some rows do not sum to 1");
        }
        write_bif(options.bif_output, bn, best_theta);
        double single_missing_pct =
            data.nrows > 0 ? 100.0 * static_cast<double>(data.single_missing_rows) / data.nrows : 0.0;
        std::ostringstream summary;
        summary << "rows=" << data.nrows << " single_missing=" << std::fixed << std::setprecision(2)
                << single_missing_pct << "% iters=" << best_iters_run
                << " best_alpha=" << best_alpha
                << " loglik=" << std::setprecision(6) << best_loglik << "\n";
        std::cerr << summary.str();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
