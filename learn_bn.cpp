#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
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

/*
Soft-EM learner for a fixed Bayesian Network structure.

Summary:
  * Parse BN structure from input BIF (variables, states, parents).
  * Load data with missing values marked as -1.
  * For each alpha in {0.1, 0.2, ..., 1.0}:
      - Initialize CPTs from complete rows (falls back to uniform if needed).
      - Run soft EM for up to 10 iterations with Dirichlet smoothing alpha.
      - Track log-likelihood and keep the best alpha by log-likelihood.
  * Write learned CPTs to the output BIF.
*/

#ifdef _OPENMP
#include <omp.h>
#endif

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
};

struct BN {
    std::vector<Variable> vars;
    std::unordered_map<std::string, int> name_to_index;
    size_t total_cpt_entries = 0;
    size_t max_states = 0;
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
    skipBlock(tz);
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

struct ThreadContext {
    std::vector<double> counts;
    std::vector<double> scratch;
};

int resolve_threads(int requested) {
#ifdef _OPENMP
    if (requested <= 0) {
        return omp_get_max_threads();
    }
    return std::max(1, requested);
#else
    (void)requested;
    return 1;
#endif
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

// Build Markov blanket node lists for each variable (parents, children, co-parents of children, children themselves)
std::vector<std::vector<int>> build_markov_blankets(const BN& bn) {
    size_t n = bn.vars.size();
    std::vector<std::vector<int>> mb(n);
    for (size_t i = 0; i < n; ++i) {
        const Variable& x = bn.vars[i];
        std::vector<int> nodes;
        nodes.reserve(x.parents.size() + x.children.size() * 3);
        // parents
        for (int p : x.parents) nodes.push_back(p);
        // children and co-parents
        for (int c : x.children) {
            nodes.push_back(c);
            const Variable& y = bn.vars[static_cast<size_t>(c)];
            for (int cp : y.parents) {
                if (cp != static_cast<int>(i)) nodes.push_back(cp);
            }
        }
        // deduplicate
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        mb[i] = std::move(nodes);
    }
    return mb;
}

// Compute topological order (Kahn's algorithm) using parent edges
std::vector<int> topological_order(const BN& bn) {
    size_t n = bn.vars.size();
    std::vector<int> indeg(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (int c : bn.vars[i].children) indeg[static_cast<size_t>(c)]++;
    }
    std::vector<int> q;
    q.reserve(n);
    for (size_t i = 0; i < n; ++i) if (indeg[i] == 0) q.push_back(static_cast<int>(i));
    std::vector<int> order;
    order.reserve(n);
    for (size_t head = 0; head < q.size(); ++head) {
        int u = q[head];
        order.push_back(u);
        for (int v : bn.vars[static_cast<size_t>(u)].children) {
            if (--indeg[static_cast<size_t>(v)] == 0) q.push_back(v);
        }
    }
    if (order.size() != n) {
        // Graph may have cycles (shouldn't in BN); fallback to 0..n-1
        order.clear();
        for (size_t i = 0; i < n; ++i) order.push_back(static_cast<int>(i));
    }
    return order;
}

struct Buckets {
    std::vector<size_t> complete;
    std::vector<std::vector<size_t>> per_var_sorted;
};

Buckets build_buckets(const BN& bn, const Data& data, const std::vector<std::vector<int>>& mb) {
    Buckets b;
    b.per_var_sorted.assign(bn.vars.size(), {});
    // First collect unsorted rows
    std::vector<std::vector<size_t>> per_var(bn.vars.size());
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            b.complete.push_back(row_idx);
        } else if (info.first_missing >= 0) {
            per_var[static_cast<size_t>(info.first_missing)].push_back(row_idx);
        }
    }
    // Rank each bucket by observational richness of the Markov blanket
    for (size_t vi = 0; vi < bn.vars.size(); ++vi) {
        const auto& rows = per_var[vi];
        if (rows.empty()) continue;
        std::vector<std::pair<int,int>> key_index; // (richness, index into rows)
        key_index.reserve(rows.size());
        const auto& mb_nodes = mb[vi];
        for (size_t j = 0; j < rows.size(); ++j) {
            const int32_t* row = data.row(rows[j]);
            int score = 0;
            for (int node : mb_nodes) {
                if (row[node] >= 0) ++score;
            }
            key_index.emplace_back(score, static_cast<int>(j));
        }
        std::sort(key_index.begin(), key_index.end(), [](const auto& a, const auto& b){
            return a.first > b.first; // descending richness
        });
        b.per_var_sorted[vi].reserve(rows.size());
        for (const auto& p : key_index) b.per_var_sorted[vi].push_back(rows[static_cast<size_t>(p.second)]);
    }
    return b;
}

// Hard E-step using bucketed processing and scheduling order
void hard_e_step_bucketed(const BN& bn, const Data& data, const std::vector<double>& theta,
                          const Buckets& buckets, const std::vector<int>& visit_order,
                          std::vector<double>& counts, int threads) {
    const size_t total = bn.total_cpt_entries;
    threads = resolve_threads(threads);
    std::vector<ThreadContext> contexts(static_cast<size_t>(threads));
    for (auto& ctx : contexts) {
        ctx.counts.assign(total, 0.0);
        ctx.scratch.assign(bn.max_states, 0.0);
    }
    // Process fully-observed rows first
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t k = 0; k < buckets.complete.size(); ++k) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
        const int32_t* row = data.row(buckets.complete[k]);
        accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
    }
    // Then for each variable bucket in the chosen order
    for (int v : visit_order) {
        const auto& rows = buckets.per_var_sorted[static_cast<size_t>(v)];
        if (rows.empty()) continue;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t k = 0; k < rows.size(); ++k) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
            const int32_t* row = data.row(rows[k]);
            const Variable& target = bn.vars[static_cast<size_t>(v)];
            bool parent_ok = true;
            size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
            if (!parent_ok) {
                accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
                continue;
            }
            double best_score = -std::numeric_limits<double>::infinity();
            int best_state = 0;
            for (int x = 0; x < target.r; ++x) {
                double log_score = safe_log(theta[target.cpt_offset + parent_idx * target.r + x]);
                for (int child_idx : target.children) {
                    const Variable& child = bn.vars[static_cast<size_t>(child_idx)];
                    int32_t child_val = row[child_idx];
                    if (child_val < 0) continue;
                    bool child_ok = true;
                    size_t child_parent_idx = 0;
                    for (size_t j = 0; j < child.parents.size(); ++j) {
                        int parent = child.parents[j];
                        int32_t val = row[parent];
                        if (parent == v) val = x;
                        if (val < 0) { child_ok = false; break; }
                        child_parent_idx += child.parent_strides[j] * static_cast<size_t>(val);
                    }
                    if (!child_ok) continue;
                    double prob = theta[child.cpt_offset + child_parent_idx * static_cast<size_t>(child.r) + static_cast<size_t>(child_val)];
                    log_score += safe_log(prob);
                }
                if (log_score > best_score) { best_score = log_score; best_state = x; }
            }
            accumulate_complete_row(bn, row, ctx.counts, 1.0, v, best_state);
        }
    }
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const auto& ctx : contexts) {
        for (size_t i = 0; i < total; ++i) counts[i] += ctx.counts[i];
    }
}

// Soft E-step bucketed
void soft_e_step_bucketed(const BN& bn, const Data& data, const std::vector<double>& theta,
                          const Buckets& buckets, const std::vector<int>& visit_order,
                          std::vector<double>& counts, int threads) {
    const size_t total = bn.total_cpt_entries;
    threads = resolve_threads(threads);
    std::vector<ThreadContext> contexts(static_cast<size_t>(threads));
    for (auto& ctx : contexts) {
        ctx.counts.assign(total, 0.0);
        ctx.scratch.assign(bn.max_states, 0.0);
    }
    // Complete rows
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t k = 0; k < buckets.complete.size(); ++k) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
        const int32_t* row = data.row(buckets.complete[k]);
        accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
    }
    // Buckets
    for (int v : visit_order) {
        const auto& rows = buckets.per_var_sorted[static_cast<size_t>(v)];
        if (rows.empty()) continue;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t k = 0; k < rows.size(); ++k) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
            const int32_t* row = data.row(rows[k]);
            const Variable& target = bn.vars[static_cast<size_t>(v)];
            bool parent_ok = true;
            size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
            if (!parent_ok) {
                accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
                continue;
            }
            double max_log = -std::numeric_limits<double>::infinity();
            for (int x = 0; x < target.r; ++x) {
                double log_score = safe_log(theta[target.cpt_offset + parent_idx * target.r + x]);
                for (int child_idx : target.children) {
                    const Variable& child = bn.vars[static_cast<size_t>(child_idx)];
                    int32_t child_val = row[child_idx];
                    if (child_val < 0) continue;
                    bool child_ok = true;
                    size_t child_parent_idx = 0;
                    for (size_t j = 0; j < child.parents.size(); ++j) {
                        int parent = child.parents[j];
                        int32_t val = row[parent];
                        if (parent == v) val = x;
                        if (val < 0) { child_ok = false; break; }
                        child_parent_idx += child.parent_strides[j] * static_cast<size_t>(val);
                    }
                    if (!child_ok) continue;
                    double prob = theta[child.cpt_offset + child_parent_idx * static_cast<size_t>(child.r) + static_cast<size_t>(child_val)];
                    log_score += safe_log(prob);
                }
                ctx.scratch[static_cast<size_t>(x)] = log_score;
                if (log_score > max_log) max_log = log_score;
            }
            double sum = 0.0;
            for (int x = 0; x < target.r; ++x) {
                ctx.scratch[static_cast<size_t>(x)] = std::exp(ctx.scratch[static_cast<size_t>(x)] - max_log);
                sum += ctx.scratch[static_cast<size_t>(x)];
            }
            if (sum <= 0.0) {
                accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
                continue;
            }
            for (int x = 0; x < target.r; ++x) {
                double w = ctx.scratch[static_cast<size_t>(x)] / sum;
                accumulate_complete_row(bn, row, ctx.counts, w, v, x);
            }
        }
    }
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const auto& ctx : contexts) {
        for (size_t i = 0; i < total; ++i) counts[i] += ctx.counts[i];
    }
}

void hard_e_step(const BN& bn, const Data& data, const std::vector<double>& theta,
                 std::vector<double>& counts, int threads) {
    const size_t total = bn.total_cpt_entries;
    threads = resolve_threads(threads);
    std::vector<ThreadContext> contexts(static_cast<size_t>(threads));
    for (auto& ctx : contexts) {
        ctx.counts.assign(total, 0.0);
        ctx.scratch.assign(bn.max_states, 0.0);
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
        std::fill(ctx.scratch.begin(), ctx.scratch.end(), -std::numeric_limits<double>::infinity());
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
            continue;
        }
        if (info.first_missing < 0) {
            continue;
        }
        const Variable& target = bn.vars[static_cast<size_t>(info.first_missing)];
        bool parent_ok = true;
        size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
        if (!parent_ok) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
            continue;
        }
        double best_score = -std::numeric_limits<double>::infinity();
        int best_state = 0;
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
            ctx.scratch[static_cast<size_t>(x)] = log_score;
            if (log_score > best_score) {
                best_score = log_score;
                best_state = x;
            }
        }
        accumulate_complete_row(bn, row, ctx.counts, 1.0, info.first_missing, best_state);
    }
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const auto& ctx : contexts) {
        for (size_t i = 0; i < total; ++i) {
            counts[i] += ctx.counts[i];
        }
    }
}

void soft_e_step(const BN& bn, const Data& data, const std::vector<double>& theta,
                 std::vector<double>& counts, int threads) {
    const size_t total = bn.total_cpt_entries;
    threads = resolve_threads(threads);
    std::vector<ThreadContext> contexts(static_cast<size_t>(threads));
    for (auto& ctx : contexts) {
        ctx.counts.assign(total, 0.0);
        ctx.scratch.assign(bn.max_states, 0.0);
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        ThreadContext& ctx = contexts[static_cast<size_t>(tid)];
        const int32_t* row = data.row(row_idx);
        RowInfo info = analyze_row(row, data.nvars);
        if (info.missing_count == 0) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
            continue;
        }
        if (info.first_missing < 0) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
            continue;
        }
        const Variable& target = bn.vars[static_cast<size_t>(info.first_missing)];
        bool parent_ok = true;
        size_t parent_idx = family_index(target, row, bn.vars, -1, -1, parent_ok);
        if (!parent_ok) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
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
            ctx.scratch[static_cast<size_t>(x)] = log_score;
            if (log_score > max_log) {
                max_log = log_score;
            }
        }
        double sum = 0.0;
        for (int x = 0; x < target.r; ++x) {
            ctx.scratch[static_cast<size_t>(x)] = std::exp(ctx.scratch[static_cast<size_t>(x)] - max_log);
            sum += ctx.scratch[static_cast<size_t>(x)];
        }
        if (sum <= 0.0) {
            accumulate_complete_row(bn, row, ctx.counts, 1.0, -1, -1);
            continue;
        }
        for (int x = 0; x < target.r; ++x) {
            double weight = ctx.scratch[static_cast<size_t>(x)] / sum;
            accumulate_complete_row(bn, row, ctx.counts, weight, info.first_missing, x);
        }
    }
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const auto& ctx : contexts) {
        for (size_t i = 0; i < total; ++i) {
            counts[i] += ctx.counts[i];
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
    for (size_t vidx = 0; vidx < bn.vars.size(); ++vidx) {
        const Variable& var = bn.vars[vidx];
        double alpha = 0.0;
        double alpha_sum = 0.0;
        double alpha_cell = 0.0;
        if (use_bdeu) {
            double ess = (user_ess > 0.0) ? user_ess : 5.0; // default ESS
            alpha_cell = ess / (static_cast<double>(var.r) * static_cast<double>(var.q));
            alpha_sum = alpha_cell * static_cast<double>(var.r);
        } else {
            alpha = effective_alpha(var, user_alpha);
            alpha_sum = alpha * static_cast<double>(var.r);
        }
        for (size_t u = 0; u < var.q; ++u) {
            size_t base = var.cpt_offset + u * static_cast<size_t>(var.r);
            double total = 0.0;
            for (int s = 0; s < var.r; ++s) {
                total += counts[base + static_cast<size_t>(s)];
            }
            double denom = total + alpha_sum;
            if (denom <= 0.0) {
                double uniform = 1.0 / static_cast<double>(var.r);
                for (int s = 0; s < var.r; ++s) {
                    theta[base + static_cast<size_t>(s)] = uniform;
                }
                continue;
            }
            for (int s = 0; s < var.r; ++s) {
                double numerator = counts[base + static_cast<size_t>(s)] + (use_bdeu ? alpha_cell : alpha);
                theta[base + static_cast<size_t>(s)] = numerator / denom;
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
                      int threads) {
    double total_loglik = 0.0;
    threads = resolve_threads(threads);
    std::vector<std::vector<double>> scratch(static_cast<size_t>(threads),
                                             std::vector<double>(bn.max_states, 0.0));
#pragma omp parallel for reduction(+ : total_loglik) schedule(static) num_threads(threads)
    for (size_t row_idx = 0; row_idx < data.nrows; ++row_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::vector<double>& log_scores = scratch[static_cast<size_t>(tid)];
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
