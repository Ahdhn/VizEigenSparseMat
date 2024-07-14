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



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Read the data from the file
data = np.loadtxt('test_mat.txt')

# Extract rows, columns, and values
rows, cols, values = data[:, 0], data[:, 1], data[:, 2]

# Visualize the sparse matrix
plt.figure(figsize=(8, 8))
plt.scatter(cols, rows, c=values, s=100, cmap='viridis', marker='s')
plt.gca().invert_yaxis()
plt.colorbar(label='Value')
plt.title('Sparse Matrix Visualization')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
# Set the x-axis and y-axis to display only integer values
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

#plt.grid(True)
plt.savefig('sparse_matrix.png', dpi=300)
plt.show()