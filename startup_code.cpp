#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <iomanip>
#include <cmath>

using namespace std;

class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = name;
        nvalues = n;
        values = vals;
    }

    string get_name() {
        return Node_Name;
    }

    vector<int> get_children() {
        return Children;
    }

    vector<string> get_Parents() {
        return Parents;
    }

    vector<float> get_CPT() {
        return CPT;
    }

    int get_nvalues() {
        return nvalues;
    }

    vector<string> get_values() {
        return values;
    }

    void set_CPT(vector<float> new_CPT) {
        CPT.clear();
        CPT = new_CPT;
    }

    void set_Parents(vector<string> Parent_Nodes) {
        Parents.clear();
        Parents = Parent_Nodes;
    }

    int add_child(int new_child_index) {
        for(int i = 0; i < Children.size(); i++) {
            if(Children[i] == new_child_index)
                return 0;
        }
        Children.push_back(new_child_index);
        return 1;
    }
};

class network {
    list<Graph_Node> Pres_Graph;

public:
    int addNode(Graph_Node node) {
        Pres_Graph.push_back(node);
        return 0;
    }

    list<Graph_Node>::iterator getNode(int i) {
        int count = 0;
        list<Graph_Node>::iterator listIt;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(count++ == i)
                break;
        }
        return listIt;
    }

    int netSize() {
        return Pres_Graph.size();
    }

    int get_index(string val_name) {
        list<Graph_Node>::iterator listIt;
        int count = 0;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(listIt->get_name().compare(val_name) == 0)
                return count;
            count++;
        }
        return -1;
    }

    list<Graph_Node>::iterator get_nth_node(int n) {
        list<Graph_Node>::iterator listIt;
        int count = 0;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(count == n)
                return listIt;
            count++;
        }
        return listIt;
    }

    list<Graph_Node>::iterator search_node(string val_name) {
        list<Graph_Node>::iterator listIt;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(listIt->get_name().compare(val_name) == 0)
                return listIt;
        }
        cout << "node not found: " << val_name << "\n";
        return listIt;
    }
};

// Helper function to trim whitespace
string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

network read_network(const char* filename) {
    network BayesNet;
    string line;
    ifstream myfile(filename);
    
    if (!myfile.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return BayesNet;
    }

    while (getline(myfile, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        stringstream ss(line);
        string token;
        ss >> token;

        // Parse variable declarations
        if (token == "variable") {
            string var_name;
            ss >> var_name;
            
            // Read the next line with type and values
            getline(myfile, line);
            stringstream ss2(line);
            
            string type_keyword, discrete_keyword, bracket, equals;
            int num_values;
            ss2 >> type_keyword >> discrete_keyword >> bracket >> num_values >> bracket >> equals >> bracket;
            
            // Read values within brackets
            vector<string> values;
            string value;
            while (ss2 >> value) {
                if (value == "};") break;
                // Remove trailing comma if present
                if (value.back() == ',') {
                    value = value.substr(0, value.length() - 1);
                }
                values.push_back(value);
            }
            
            Graph_Node new_node(var_name, num_values, values);
            BayesNet.addNode(new_node);
        }
        // Parse probability tables
        else if (token == "probability") {
            string paren, node_name;
            ss >> paren >> node_name;
            
            // Handle multi-line probability declarations
            string full_prob_line = line;
            while (full_prob_line.find('{') == string::npos) {
                string next_line;
                if (!getline(myfile, next_line)) break;
                full_prob_line += " " + trim(next_line);
            }
            
            // Extract node name and parents from the probability declaration
            size_t start_paren = full_prob_line.find('(');
            size_t end_paren = full_prob_line.find(')');
            size_t pipe_pos = full_prob_line.find('|');
            
            if (start_paren == string::npos || end_paren == string::npos) {
                continue;
            }
            
            string prob_content = full_prob_line.substr(start_paren + 1, end_paren - start_paren - 1);
            stringstream prob_ss(prob_content);
            
            prob_ss >> node_name;
            
            list<Graph_Node>::iterator listIt = BayesNet.search_node(node_name);
            int index = BayesNet.get_index(node_name);
            
            vector<string> parents;
            
            // Check if there are parent nodes (conditional probability)
            if (pipe_pos != string::npos && pipe_pos < end_paren) {
                string parents_str = full_prob_line.substr(pipe_pos + 1, end_paren - pipe_pos - 1);
                stringstream parent_ss(parents_str);
                string parent;
                
                while (parent_ss >> parent) {
                    // Remove trailing comma if present
                    if (parent.back() == ',') {
                        parent = parent.substr(0, parent.length() - 1);
                    }
                    parents.push_back(parent);
                    
                    list<Graph_Node>::iterator parentIt = BayesNet.search_node(parent);
                    parentIt->add_child(index);
                }
            }
            
            listIt->set_Parents(parents);
            
            // Read CPT values - everything between { and };
            vector<float> cpt;
            bool reading_cpt = false;
            
            while (getline(myfile, line)) {
                line = trim(line);
                if (line == "};") break;
                if (line.empty()) continue;
                
                // Parse the line for probability values
                size_t close_paren = line.find(')');
                string prob_part;
                
                if (close_paren != string::npos) {
                    // Line has conditional values like "(A) 0.5, 0.5;"
                    prob_part = line.substr(close_paren + 1);
                } else if (line.find("table") != string::npos) {
                    // Line has "table" keyword
                    size_t table_pos = line.find("table");
                    prob_part = line.substr(table_pos + 5);
                } else {
                    prob_part = line;
                }
                
                // Extract all probability values from the line
                stringstream ss_prob(prob_part);
                string token;
                while (ss_prob >> token) {
                    // Remove trailing punctuation
                    while (!token.empty() && (token.back() == ',' || token.back() == ';')) {
                        token = token.substr(0, token.length() - 1);
                    }
                    
                    // Check if it's a valid number
                    if (!token.empty() && (isdigit(token[0]) || token[0] == '.' || token[0] == '-')) {
                        cpt.push_back(atof(token.c_str()));
                    }
                }
            }
            
            listIt->set_CPT(cpt);
        }
    }
    
    myfile.close();
    return BayesNet;
}


void write_network(const char* filename, network& BayesNet) {
    ofstream outfile(filename);

    if (!outfile.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }

    outfile << "// Bayesian Network" << endl << endl;

    int N = BayesNet.netSize();

    // Write all nodes first
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);

        outfile << "variable " << node->get_name() << " {" << endl;
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";

        vector<string> vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++) {
            outfile << vals[j];
            if (j < (int)vals.size() - 1) outfile << ", ";
        }
        outfile << " };" << endl;
        outfile << "}" << endl;
    }

    // Write probability tables
    outfile << std::fixed << std::setprecision(6); // consistent float formatting
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);
        vector<string> parents = node->get_Parents();
        vector<string> values = node->get_values();
        vector<float> cpt = node->get_CPT();

        // Header
        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++) {
                outfile << parents[j];
                if (j < (int)parents.size() - 1) outfile << ", ";
            }
        }
        outfile << " ) {" << endl;

        // compute parent radices (number of values for each parent)
        vector<int> radices;
        radices.reserve(parents.size());
        for (auto &pname : parents) {
            auto pnode = BayesNet.search_node(pname);
            radices.push_back(pnode->get_nvalues());
        }

        // product of radices
        int parent_combinations = 1;
        for (int r : radices) parent_combinations *= r;

        // Write the CPT values. Keep a single cpt_index that we advance in the same order read_network used
        int cpt_index = 0;

        if (parents.empty()) {
            // unconditional: use table
            outfile << "    table ";
            for (int k = 0; k < (int)values.size(); k++) {
                if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++];
                else outfile << "-1";
                if (k < (int)values.size() - 1) outfile << ", ";
            }
            outfile << ";" << endl;
        } else {
            // conditional: for each parent combination write "( v1, v2, ... ) p1, p2, ...;"
            for (int comb = 0; comb < parent_combinations; comb++) {
                // compute mixed-radix digits (indices for each parent),
                // with the rightmost parent cycling fastest (so we fill indices from right->left)
                vector<int> idx(parents.size(), 0);
                int tmp = comb;
                for (int p = (int)parents.size() - 1; p >= 0; p--) {
                    idx[p] = tmp % radices[p];
                    tmp /= radices[p];
                }

                // print parent values in the original parent order
                outfile << "    ( ";
                for (int p = 0; p < (int)parents.size(); p++) {
                    auto pnode = BayesNet.search_node(parents[p]);
                    auto pvals = pnode->get_values();
                    int vidx = idx[p];
                    outfile << pvals[vidx];
                    if (p < (int)parents.size() - 1) outfile << ", ";
                }
                outfile << " ) ";

                // print the probabilities for this parent combination (one probability per child value)
                for (int k = 0; k < (int)values.size(); k++) {
                    if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++]; // round to last non zero decimal valu
                    else outfile << "-1";
                    if (k < (int)values.size() - 1) outfile << ", ";
                }
                outfile << ";" << endl;
            }
        }

        // ***** THIS IS THE FIX *****
        outfile << "};" << endl << endl;
    }

    outfile.close();
    cout << "Network written to file: " << filename << endl;
}






#ifndef BN_LIB
int main() {
    network BayesNet = read_network("solved.bif");
    
    cout << "Network loaded successfully!" << endl;
    cout << "Number of nodes: " << BayesNet.netSize() << endl;
    
    return 0;
}
#endif // BN_LIB