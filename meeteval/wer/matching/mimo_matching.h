#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>


void update_levenshtein_row(
        std::vector<unsigned int> &row,
        const std::vector<unsigned int> &reference,
        const std::vector<unsigned int> &hypothesis
) {
    // TODO: track where stuff came from
    // Temporary variables
    unsigned int up;    // Value above the current cell (deletion)
    unsigned int diagonal;  // Value diagonal to the current cell (substitution)
    unsigned int left;

    for (auto ref_symbol: reference) {
        diagonal = row[0]++;  // diagonal = row[0]; row[0] = row[0] + 1;
        left = row[0];

        unsigned int hyp_index = 1;
        for (auto hyp_symbol: hypothesis) {
            up = row[hyp_index];
            if (ref_symbol == hyp_symbol) {
                left = diagonal;
            } else {
                left = 1 + std::min(std::min(left, up), diagonal);
            }
            row[hyp_index] = left;
            diagonal = up;
            hyp_index++;
        }
    }
}

unsigned int levenshtein_distance_(
        const std::vector<unsigned int> reference,
        const std::vector<unsigned int> hypothesis
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1);

    // Initialize with range
    std::iota(row.begin(), row.end(), 0);

    update_levenshtein_row(row, reference, hypothesis);

    return row[row.size() - 1];
}

struct Layout {
    std::vector<unsigned int> strides;
    std::vector<unsigned int> dimensions;
    unsigned int total_size;
};

template<typename T>
Layout inline make_layout(const std::vector<std::vector<T>> vec) {
    Layout layout;
    for (auto v : vec) layout.dimensions.push_back(v.size() + 1);
    unsigned int j = 1;
    for (auto i : layout.dimensions) {
        layout.strides.push_back(j);
        j *= i;
    }
    layout.total_size = j;
    return layout;
}

unsigned int mimo_matching_(
        std::vector<std::vector<std::vector<unsigned int>>> references, std::vector<std::vector<unsigned int>> hypotheses
) {
    // Get dimensions and layouts for the storage
    Layout ref_layout = make_layout(references);
    Layout hyp_layout = make_layout(hypotheses);

    // Build reference storage grid
    std::vector<std::vector<unsigned int>> ref_grid(ref_layout.total_size);

    // Initialize first element
    auto state = std::vector<unsigned int>(hyp_layout.total_size);
    for (unsigned int i = 0; i < hyp_layout.total_size; i++) {
        unsigned int j = 0;
        // TODO: can this be simplified?
        for (unsigned int v = 0; v < hyp_layout.dimensions.size(); v++) {
            j += (i / hyp_layout.strides[v]) % hyp_layout.dimensions[v];
        }
        state[i] = j;
    }
    ref_grid[0] = state;

    std::vector<unsigned int> reference_indices(references.size(), 0);

    // Main loop
    for (unsigned int reference_utterance_index = 1; reference_utterance_index < ref_layout.total_size; reference_utterance_index++) {
        // Advance the index by one
        for (unsigned int i = 0; i < references.size(); i++) {
            reference_indices[i]++;
            if (reference_indices[i] < ref_layout.dimensions[i]) break;
            reference_indices[i] = 0;
        }

        // Get destination storage
        auto state = std::vector<unsigned int>(hyp_layout.total_size);
        auto first_update = true;

        // Loop through all adjacent reference combinations and advance row
        for (unsigned int active_reference_index = 0; active_reference_index < reference_indices.size(); active_reference_index++) {
            if (reference_indices[active_reference_index] == 0) continue;

            // Get the previous state, only based on the current assignment of
            // references
            std::vector<unsigned int> &previous_state = ref_grid[reference_utterance_index - ref_layout.strides[active_reference_index]];
            std::vector<unsigned int> &active_reference = references[active_reference_index][reference_indices[active_reference_index] - 1];

            // Forward levenshtein row
            for (unsigned int active_hypothesis_index = 0; active_hypothesis_index < hypotheses.size(); active_hypothesis_index++) {
                for (
                        unsigned int _i = 0;
                        _i < hyp_layout.total_size;
                        _i += hyp_layout.strides.size() > active_hypothesis_index + 1 ? hyp_layout.strides[active_hypothesis_index + 1] : hyp_layout.total_size
                ) {
                    for (
                            unsigned int i = _i;
                            i < _i + hyp_layout.strides[active_hypothesis_index];
                            i++
                    ) {
                        // Copy the levenshtein row into the buffer
                        std::vector<unsigned int> tmp_row(hyp_layout.dimensions[active_hypothesis_index]);
                        for (unsigned int j = 0; j < hyp_layout.dimensions[active_hypothesis_index]; j++) {
                            tmp_row[j] = previous_state[i + j * hyp_layout.strides[active_hypothesis_index]];
                        }

                        // Apply levenshtein algorithm
                        update_levenshtein_row(tmp_row, active_reference, hypotheses[active_hypothesis_index]);

                        // Update current state. Keep only the min value. Do inverse of the above stride access
                        for (unsigned int j = 0; j < hyp_layout.dimensions[active_hypothesis_index]; j++) {
                            if (first_update || state[i + j * hyp_layout.strides[active_hypothesis_index]] > tmp_row[j]) {
                                state[i + j*hyp_layout.strides[active_hypothesis_index]] = tmp_row[j];
                            }
                        }
                    }
                }
                first_update = false;
            }
        }
        ref_grid[reference_utterance_index] = state;
    }
    return ref_grid.back().back();
}

#pragma clang diagnostic pop