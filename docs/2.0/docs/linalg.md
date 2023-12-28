# torch.linalg [¶](#torch-linalg "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/linalg>
>
> 原始地址：<https://pytorch.org/docs/stable/linalg.html>


 常见的线性代数运算。


 有关一些常见的数值边缘情况，请参阅[线性代数 (torch.linalg)](notes/numerical_accuracy.html#linear-algebra-stability)。


## 矩阵属性 [¶](#matrix-properties "此标题的固定链接")


|  |  |
| --- | --- |
| [`norm`](generated/torch.linalg.norm.html#torch.linalg.norm "torch.linalg.norm") |计算向量或矩阵范数。 |
| [`vector_norm`](generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm "torch.linalg.vector_norm") |计算向量范数。 |
| [`matrix_norm`](generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm "torch.linalg.matrix_norm") |计算矩阵范数。 |
| [`对角线`](generated/torch.linalg.diagonal.html#torch.linalg.diagonal "torch.linalg.diagonal") | [`torch.diagonal()`](generated/torch.diagonal.html#torch.diagonal "torch.diagonal") 的别名，默认值 `dim1` = -2 、 `dim2` = -1 。 |
| [`it`](generated/torch.linalg.det.html#torch.linalg.det "torch.linalg.det") |计算方阵的行列式。 |
| [`slogdet`](generated/torch.linalg.slogdet.html#torch.linalg.slogdet "torch.linalg.slogdet") |计算方阵行列式的绝对值的符号和自然对数。 |
| [`cond`](generated/torch.linalg.cond.html#torch.linalg.cond“torch.linalg.cond”) |计算矩阵相对于矩阵范数的条件数。 |
| [`matrix_rank`](generated/torch.linalg.matrix_rank.html#torch.linalg.matrix_rank "torch.linalg.matrix_rank") |计算矩阵的数值秩。 |


## 分解 [¶](#decompositions "此标题的永久链接")


|  |  |
| --- | --- |
| [`cholesky`](generated/torch.linalg.cholesky.html#torch.linalg.cholesky "torch.linalg.cholesky") |计算复数埃尔米特矩阵或实数对称正定矩阵的 Cholesky 分解。 |
| [`qr`](generated/torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr") |计算矩阵的 QR 分解。 |
| [`lu`](generated/torch.linalg.lu.html#torch.linalg.lu "torch.linalg.lu") |通过矩阵的部分旋转计算 LU 分解。 |
| [`lu_factor`](generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor") |通过矩阵的部分旋转计算 LU 分解的紧凑表示。 |
| [`eig`](generated/torch.linalg.eig.html#torch.linalg.eig "torch.linalg.eig") |计算方阵的特征值分解(如果存在)。 |
| [`eigvals`](generated/torch.linalg.eigvals.html#torch.linalg.eigvals "torch.linalg.eigvals") |计算方阵的特征值。 |
| [`eigh`](generated/torch.linalg.eigh.html#torch.linalg.eigh "torch.linalg.eigh") |计算复数埃尔米特矩阵或实对称矩阵的特征值分解。 |
| [`eigvalsh`](generated/torch.linalg.eigvalsh.html#torch.linalg.eigvalsh "torch.linalg.eigvalsh") |计算复数埃尔米特矩阵或实对称矩阵的特征值。 |
| [`svd`](generated/torch.linalg.svd.html#torch.linalg.svd "torch.linalg.svd") |计算矩阵的奇异值分解 (SVD)。 |
| [`svdvals`](generated/torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals") |计算矩阵的奇异值。 |


## 求解器 [¶](#solvers "此标题的永久链接")


|  |  |
| --- | --- |
| [`解决`](generated/torch.linalg.solve.html#torch.linalg.solve“torch.linalg.solve”)|计算具有唯一解的平方线性方程组的解。 |
| [`solve_triangle`](generated/torch.linalg.solve_triangle.html#torch.linalg.solve_triangle "torch.linalg.solve_triangle") |计算具有唯一解的三角线性方程组的解。 |
| [`lu_solve`](generated/torch.linalg.lu_solve.html#torch.linalg.lu_solve "torch.linalg.lu_solve") |在给定 LU 分解的情况下，计算具有唯一解的平方线性方程组的解。 |
| [`lstsq`](generated/torch.linalg.lstsq.html#torch.linalg.lstsq "torch.linalg.lstsq") |计算线性方程组最小二乘问题的解。 |


## Inverses [¶](#inverses "此标题的永久链接")


|  |  |
| --- | --- |
| [`inv`](generated/torch.linalg.inv.html#torch.linalg.inv "torch.linalg.inv") |计算方阵的逆矩阵(如果存在)。 |
| [`pinv`](generated/torch.linalg.pinv.html#torch.linalg.pinv "torch.linalg.pinv") |计算矩阵的伪逆(Moore-Penrose 逆)。 |


## 矩阵函数 [¶](#matrix-functions "此标题的固定链接")


|  |  |
| --- | --- |
| [`matrix_exp`](generated/torch.linalg.matrix_exp.html#torch.linalg.matrix_exp "torch.linalg.matrix_exp") |计算方阵的矩阵指数。 |
| [`matrix_power`](generated/torch.linalg.matrix_power.html#torch.linalg.matrix_power "torch.linalg.matrix_power") |计算整数 n 的方阵的 n 次方。 |


## 矩阵产品[¶](#matrix-products"此标题的固定链接")


|  |  |
| --- | --- |
| [`cross`](generated/torch.linalg.cross.html#torch.linalg.cross "torch.linalg.cross") |计算两个 3 维向量的叉积。 |
| [`matmul`](generated/torch.linalg.matmul.html#torch.linalg.matmul "torch.linalg.matmul") | [`torch.matmul()`](generated/torch.matmul.html#torch.matmul "torch.matmul") 的别名 |
| [`vecdot`](generated/torch.linalg.vecdot.html#torch.linalg.vecdot“torch.linalg.vecdot”) |计算沿某个维度的两批向量的点积。 |
| [`multi_dot`](generated/torch.linalg.multi_dot.html#torch.linalg.multi_dot "torch.linalg.multi_dot") |通过重新排序乘法来有效地乘法两个或多个矩阵，以便执行最少的算术运算。 |
| [`householder_product`](generated/torch.linalg.householder_product.html#torch.linalg.householder_product "torch.linalg.householder_product") |计算 Householder 矩阵乘积的前 n 列。 |


## tensor运算 [¶](#tensor-operations "此标题的固定链接")


|  |  |
| --- | --- |
| [`tensorinv`](generated/torch.linalg.tensorinv.html#torch.linalg.tensorinv "torch.linalg.tensorinv") |计算 [`torch.tensordot()`](generated/torch.tensordot.html#torch.tensordot "torch.tensordot") 的乘法逆元。 |
| [`tensorsolve`](generated/torch.linalg.tensorsolve.html#torch.linalg.tensorsolve "torch.linalg.tensorsolve") |计算系统 torch.tensordot(A, X) = B 的解 X 。 |


## Misc [¶](#misc "此标题的永久链接")


|  |  |
| --- | --- |
| [`vander`](generated/torch.linalg.vander.html#torch.linalg.vander "torch.linalg.vander") |生成范德蒙矩阵。 |


## 实验函数 [¶](#experimental-functions "此标题的永久链接")


|  |  |
| --- | --- |
| [`cholesky_ex`](generated/torch.linalg.cholesky_ex.html#torch.linalg.cholesky_ex "torch.linalg.cholesky_ex") |计算复数埃尔米特矩阵或实数对称正定矩阵的 Cholesky 分解。 |
| [`inv_ex`](generated/torch.linalg.inv_ex.html#torch.linalg.inv_ex "torch.linalg.inv_ex") |如果方阵可逆，则计算方阵的逆。 |
| [`solve_ex`](generated/torch.linalg.solve_ex.html#torch.linalg.solve_ex "torch.linalg.solve_ex") | [`solve()`](generated/torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve") 的一个版本，除非 `check_errors` = True ，否则不会执行错误检查。 |
| [`lu_factor_ex`](generated/torch.linalg.lu_factor_ex.html#torch.linalg.lu_factor_ex "torch.linalg.lu_factor_ex") |这是 [`lu_factor()`](generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor") 的一个版本，除非 `check_errors` ，否则不会执行错误检查= 正确。 |
| [`ldl_factor`](generated/torch.linalg.ldl_factor.html#torch.linalg.ldl_factor "torch.linalg.ldl_factor") |计算 Hermitian 或对称(可能不定)矩阵的 LDL 分解的紧凑表示。 |
| [`ldl_factor_ex`](generated/torch.linalg.ldl_factor_ex.html#torch.linalg.ldl_factor_ex "torch.linalg.ldl_factor_ex") |这是 [`ldl_factor()`](generated/torch.linalg.ldl_factor.html#torch.linalg.ldl_factor "torch.linalg.ldl_factor") 的一个版本，除非 `check_errors`，否则不会执行错误检查= 正确。 |
| [`ldl_solve`](generated/torch.linalg.ldl_solve.html#torch.linalg.ldl_solve "torch.linalg.ldl_solve") |使用 LDL 分解计算线性方程组的解。 |