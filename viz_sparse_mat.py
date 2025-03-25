#C++ code to save Eigen SparseMatrix
#
#template<typename T>
#void save_sparse_mat(const Eigen::SparseMatrix<T>& mat, const std::string& filename) {
#    std::ofstream file(filename);
#    for (int k = 0; k < mat.outerSize(); ++k) {
#        for (Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
#            file << it.row() << " " << it.col() << " " << it.value() << "\n";
#        }
#    }
#    file.close();
#}


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Read the data from the file
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <filename>")
    sys.exit(1)

# Read the filename from the command-line argument
filename = sys.argv[1]

data = np.loadtxt(filename)

# Extract rows, columns, and values
rows, cols, values = data[:, 0], data[:, 1], data[:, 2]


## Calculate the number of rows
#num_rows = int(rows.max()) + 1  # Assuming row indices start from 0
#
## Initialize an array to count non-zero entries per row
#nnz_per_row = np.zeros(num_rows, dtype=int)
#
## Count non-zero elements per row
#for row in rows:
#    nnz_per_row[int(row)] += 1
#
## Print the number of non-zero elements per row
#for i, count in enumerate(nnz_per_row):
#    print(f"Row {i}: {count} non-zero elements")


num_non_zero = len(values)
num_rows = int(rows.max()) + 1
num_cols = int(cols.max()) + 1
sparsity_ratio = num_non_zero / (num_rows * num_cols)

print(f"Sparse matrix size: {num_rows} x {num_cols}")
print(f"Non-zero entries: {num_non_zero}")
print(f"Sparsity ratio: {sparsity_ratio:.6f}")
    
# Visualize the sparse matrix
plt.figure(figsize=(8, 8))

#plt.scatter(cols, rows,  c=np.ones_like(values), s=0.5, cmap='viridis', marker='s')
plt.scatter(cols, rows,  c=values, s=0.5, cmap='viridis', marker='s')

plt.gca().invert_yaxis()
#plt.colorbar(label='Value')
#plt.xlabel('Column')
#plt.ylabel('Row')
# Set the x-axis and y-axis to display only integer values
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

#plt.grid(True)
plt.savefig(filename + ".png", dpi=300)
plt.show()