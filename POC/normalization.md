### Derivation 1: Gradient of $f(x) = \frac{x}{\|x\|}$

Let $f(x) = \frac{x}{\|x\|}$, where $x \in \mathbb{R}^n$ and $x \neq 0$. The Euclidean norm is $\|x\| = \sqrt{\sum_{k=1}^n x_k^2}$.
The $i$-th component of $f(x)$ is $f_i(x) = \frac{x_i}{\|x\|}$.
The gradient $\nabla f(x)$ is the Jacobian matrix $J$ with entries $J_{ij} = \frac{\partial f_i}{\partial x_j}$.

1.  **Apply the Quotient Rule**:
    For $f_i(x) = \frac{x_i}{\|x\|}$, the partial derivative with respect to $x_j$ is:
    $$ J_{ij} = \frac{\partial}{\partial x_j}\left(\frac{x_i}{\|x\|}\right) = \frac{\left(\frac{\partial x_i}{\partial x_j}\right)\|x\| - x_i\left(\frac{\partial \|x\|}{\partial x_j}\right)}{\|x\|^2} $$

2.  **Calculate component derivatives**:
    * $\frac{\partial x_i}{\partial x_j} = \delta_{ij}$ (where $\delta_{ij}$ is the Kronecker delta: 1 if $i=j$, 0 otherwise).
    * $\frac{\partial \|x\|}{\partial x_j} = \frac{\partial}{\partial x_j}\left(\sum_{k=1}^n x_k^2\right)^{1/2} = \frac{1}{2}\left(\sum_{k=1}^n x_k^2\right)^{-1/2}(2x_j) = \frac{x_j}{\|x\|}$.

3.  **Substitute derivatives into the quotient rule expression**:
    $$ J_{ij} = \frac{\delta_{ij}\|x\| - x_i\left(\frac{x_j}{\|x\|}\right)}{\|x\|^2} = \frac{\delta_{ij}\|x\|^2 - x_i x_j}{\|x\|^3} $$

4.  **Express in matrix form**:
    The Jacobian matrix $J = \nabla f(x)$ can be written as:
    $$ \nabla\left(\frac{x}{\|x\|}\right) = \frac{I}{\|x\|} - \frac{xx^T}{\|x\|^3} $$
    where $I$ is the $n \times n$ identity matrix and $xx^T$ is the outer product of $x$ with itself.

---

### Derivation 2: Gradient of $g(x) = \frac{x}{\text{rms-norm}(x)}$

Let $g(x) = \frac{x}{\text{rms-norm}(x)}$, where $x \in \mathbb{R}^n$ and $x \neq 0$.

1.  **Define the RMS-norm**:
    The Root Mean Square (RMS) norm of $x$ is:
    $$ \text{rms-norm}(x) = \sqrt{\frac{1}{n}\sum_{k=1}^n x_k^2} = \sqrt{\frac{\|x\|^2}{n}} = \frac{\|x\|}{\sqrt{n}} $$

2.  **Rewrite the function $g(x)$**:
    Substitute the definition of the RMS-norm into $g(x)$:
    $$ g(x) = \frac{x}{\frac{\|x\|}{\sqrt{n}}} = \sqrt{n} \frac{x}{\|x\|} $$

3.  **Relate $g(x)$ to the previous function $f(x)$**:
    Notice that $g(x) = \sqrt{n} \cdot f(x)$, where $f(x) = \frac{x}{\|x\|}$ from Derivation 1.

4.  **Calculate the gradient of $g(x)$**:
    Using the property $\nabla (c \cdot h(x)) = c \cdot \nabla h(x)$ for a scalar constant $c$:
    $$ \nabla g(x) = \nabla\left(\sqrt{n} \frac{x}{\|x\|}\right) = \sqrt{n} \nabla\left(\frac{x}{\|x\|}\right) $$

5.  **Substitute the result from Derivation 1**:
    $$ \nabla g(x) = \sqrt{n} \left( \frac{I}{\|x\|} - \frac{xx^T}{\|x\|^3} \right) $$
    $$ \nabla g(x) = \frac{\sqrt{n}I}{\|x\|} - \frac{\sqrt{n}xx^T}{\|x\|^3} $$
    This is the gradient of $\frac{x}{\text{rms-norm}(x)}$. It is $\sqrt{n}$ times the gradient of $\frac{x}{\|x\|}$.