import numpy as np
import ctypes

# Размеры и параметры для примера
batch_size = 2
n = 3
nrhs = 2
coeff_matrix_elemsize = ctypes.sizeof(ctypes.c_double)
dependent_vals_elemsize = ctypes.sizeof(ctypes.c_double)

# Создаем примерные массивы NumPy
coeff_matrix = np.arange(batch_size * n * n, dtype=np.double).reshape(batch_size, n, n)
dependent_vals = np.arange(batch_size * n * nrhs, dtype=np.double).reshape(batch_size, n, nrhs)

# my logic
coeff_matrix = coeff_matrix.transpose((0, 2, 1))
dependent_vals = dependent_vals.transpose((0, 2, 1))

coeff_matrix = np.array(coeff_matrix, order="C")
dependent_vals = np.array(dependent_vals, order="C")

# # Sasha`s logic
# coeff_matrix = np.moveaxis(coeff_matrix, (-2, -1), (0, 1))
# dependent_vals = np.moveaxis(dependent_vals, (-2, -1), (0, 1))

# coeff_matrix = np.array(coeff_matrix, order="F")
# dependent_vals = np.array(dependent_vals, order="F")

# get pointer
coeff_matrix_data = coeff_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
dependent_vals_data = dependent_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Функция для получения значений по указателю с учетом смещения и размера элемента
def get_values_from_pointer(ptr, step, count, elemsize):
    result = []
    for i in range(count):
        result.append(ptr[i * step // elemsize])
    return result


for i in range(batch_size):
    coeff_matrix_batch = ctypes.cast(ctypes.addressof(coeff_matrix_data.contents) + i * n * n * coeff_matrix_elemsize, ctypes.POINTER(ctypes.c_double))
    dependent_vals_batch = ctypes.cast(ctypes.addressof(dependent_vals_data.contents) + i * n * nrhs * dependent_vals_elemsize, ctypes.POINTER(ctypes.c_double))

    # get value by new pointer
    coeff_values = get_values_from_pointer(coeff_matrix_batch, coeff_matrix_elemsize, n * n, coeff_matrix_elemsize)
    dependent_values = get_values_from_pointer(dependent_vals_batch, dependent_vals_elemsize, n * nrhs, dependent_vals_elemsize)

    print(f"Batch {i + 1}:")
    print("Coefficients:", coeff_values)
    print("Dependent values:", dependent_values)
