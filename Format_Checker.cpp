#define BN_LIB
#include "starter.cpp"   // use existing parser & data structures
#include <cmath>
#include <iomanip>

using namespace std;

float compute_error(network& gold, network& solved) {
    float total_error = 0.0;
    int num = 0;

    if (gold.netSize() != solved.netSize()) {
        cout << "Warning: Networks have different sizes!\n";
    }
    int N = min(gold.netSize(), solved.netSize());
    for (int i = 0; i < N; i++) {
        auto gold_node = gold.get_nth_node(i);
        auto solved_node = solved.get_nth_node(i);

        if (gold_node->get_name() != solved_node->get_name()) {
            cout << "Warning: Node mismatch (" 
                 << gold_node->get_name() << " vs " 
                 << solved_node->get_name() << ")\n";
            continue;
        }

        vector<float> gCPT = gold_node->get_CPT();
        vector<float> sCPT = solved_node->get_CPT();

        int len = min(gCPT.size(), sCPT.size());
        for (int j = 0; j < len; j++) {
            total_error += fabs(gCPT[j] - sCPT[j]);
            num++;
        }
    }
    return total_error/num;
}

bool check_structure(network& ref, network& test) {
    bool ok = true;

    if (ref.netSize() != test.netSize()) {
        cout << "Error: Different number of nodes (" 
             << ref.netSize() << " vs " << test.netSize() << ")\n";
        ok = false;
    }

    for (int i = 0; i < ref.netSize(); i++) {
        auto ref_node = ref.get_nth_node(i);
        auto test_node = test.get_nth_node(i);

        if (ref_node->get_name() != test_node->get_name()) {
            cout << "Node name mismatch: " << ref_node->get_name()
                 << " vs " << test_node->get_name() << endl;
            ok = false;
        }

        if (ref_node->get_Parents() != test_node->get_Parents()) {
            cout << "Parent mismatch in node: " << ref_node->get_name() << endl;
            ok = false;
        }

        if (ref_node->get_values() != test_node->get_values()) {
            cout << "Value mismatch in node: " << ref_node->get_name() << endl;
            ok = false;
        }

        // Check CPT completeness (no -1)
        for (float p : test_node->get_CPT()) {
            if (p == -1) {
                cout << "Unlearned probability in " 
                     << test_node->get_name() << endl;
                ok = false;
                break;
            }
        }

        // Check normalization (sum ~ 1)
        int num_vals = ref_node->get_nvalues();
        vector<float> cpt = test_node->get_CPT();

        if (!cpt.empty()) {
            for (size_t j = 0; j < cpt.size(); j += num_vals) {
                float s = 0;
                for (int k = 0; k < num_vals && j + k < cpt.size(); k++) {
                    s += cpt[j + k];
                }
                if (fabs(s - 1.0) > 1e-3) {
                    cout << "CPT row not normalized in " 
                         << test_node->get_name()
                         << " (sum=" << s << ")" << endl;
                    ok = false;
                }
            }
        }
    }
    return ok;
}

int main() {
    cout << "=== FORMAT CHECKER ===" << endl;

    network base = read_network("hailfinder.bif");
    network solved = read_network("solved.bif");
    network gold = read_network("gold_hailfinder.bif");

    cout << "Network loaded successfully!" << endl;
    cout << "Number of nodes in base: " << base.netSize() << endl;
    cout << "Number of nodes in solved: " << solved.netSize() << endl;
    cout << "Number of nodes in gold: " << gold.netSize() << endl;

    cout << "Files loaded successfully.\n";
    cout << "Checking structure consistency...\n";

    bool ok = check_structure(base, solved);

    if (!ok) {
        cout << "Format check failed. Please fix the structure errors above.\n";
    } else {
        cout << "Format check passed.\n";
    }

    cout << "\nComputing total learning error...\n";
    float total_error = compute_error(gold, solved);
    cout << "Total learning error: " 
         << fixed << setprecision(6) << total_error << endl;

    return 0;
}
