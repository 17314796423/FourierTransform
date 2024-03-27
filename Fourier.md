# FT

1. 套用**欧拉(旋转)公式**我们得到单位时间内顺时针转**f**圈的单位向量表达式：**exp( - 2π i f t )**
2. 对于要进行傅里叶变换的输入信号g(t)，我们做这样的变换：**g(t) * exp( - 2π i f t )**，就是转的单位向量变成了模为**g(t)**的向量(复平面里头就是一个向量)表达式，然后我们计算这个曲线的质心，如下
![image-20240327144931174](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327144931174.png)

> 这里的ds是相对于x的，而傅里叶的积分是基于t的，线长随t线性增加，那么ds=dt，ρ=1，得到如下公式，分别是连续和离散的近傅里叶变换，如果是算质量，那就不要除以线长，也就是傅里叶变换的公式。
>
> ![image-20240327150441837](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327150441837.png)
>
> ![image-20240327151814095](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327151814095.png)
3. 那么上式近似傅里叶变换的几何意义就代表将收到的信号强度在复平面内绕原点顺时针以单位时间内**f**圈的速率均匀缠绕后，计算这根线的质心位置（如果把这个被积分函数当做向量的话）

![image-20240327203612423](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327203612423.png)

4. 当**f**等于原信号频率**k**时，相当于每经过单位时间，将单位时间内收到**k**个周期的信号顺时针绕了**k**圈

5. 此时全部信号重叠，考虑曲线缠绕的物理性质，不等于**k**的时候曲线均匀分布，质心始终接在原点附近，等于**k**时全部重叠的曲线使得x方向上的质心与y方向的质心偏离原点达到最远处，那么线的总体质量也是最大的

   ![image-20240327203948116](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327203948116.png)

6. 随着绕的圈数越多（即收到信号的时长**t2-t1**越大），线的总质量也越大（即线的总质量关于**f**的函数**F(f)**的最大值随圈数线性增长）

7. 多个单频信号**gi**叠在一起时候，如果**f**等于**fi**，那么信号中只有**gi**的那部分会重叠，使得gi的那部分质心被增强

   ![image-20240327205133804](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327205133804.png)

8. 而其余频率的信号依然被均匀分布在各处，他们的质心依然在原点附近，所以线的总质量此时会被放大到**gi**对应的总质量最值

9. 对于单频信号如果波峰不一样没关系，重叠时候无非是叠在一起的总质量加成有的大有的小而已，相位重叠的，不是完全重叠而已

10. 同样的两个频率信号接收时长相差很大的情况下，在**f**等于**k**处对于频率**f**的同样的微小变动，两个曲线以同样的交错度铺开，但因为接收时间长的信号曲线更长，所以沿着同样的交错度铺的更长也就会更快铺均匀，使得质心更快回到原点附近

11. 意味着傅里叶变换后得到的波峰的宽度会非常窄和细高，反之接受时间短的信号反而在f进行k附近很大范围变动的时候质心都不会快速回到原点，

12. 也就代表波峰宽度非常大，甚至形成非常多段的波浪在原点较远处来回大幅度摆动

13. 傅立叶变换的另一个视角就是算每个频率下正余弦函数与原函数的余弦相似度：向量的内积，当然不是单位向量还包含了两个向量的模在里面

14. 二维傅立叶的几何意义就是横纵轴正余弦函数以不同频率笛卡尔积组合，得到的两个向量（离散）/曲线（连续）的张量积形成曲面与原函数曲面的内积相似度，代数表达式含义是横着全部傅立叶来一遍，再竖着来一遍，先后无所谓

# DFT

1. 

$$
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i \frac{kn}{N}} = \sum_{n=0}^{N-1} x_n \cdot \omega^{k \cdot n} \quad\text{where}\quad\omega = \text{e}^{-\frac{2\pi i}{N}}
$$

2. $$
   \begin{bmatrix}
   X_0 \\
   X_1 \\
   \vdots \\
   X_{N-1}
   \end{bmatrix}_{\underset{RESULT}{}}
   = 
   \begin{bmatrix}
   \omega^{0 \cdot 0} \quad \omega^{0 \cdot 1} \quad \cdots \quad \omega^{0 \cdot (N-1)} \\
   \omega^{1 \cdot 0} \quad \omega^{1 \cdot 1} \quad \cdots \quad \omega^{1 \cdot (N-1)} \\
   \vdots \quad \vdots \quad \ddots \quad \vdots \\
   \omega^{(N-1) \cdot 0} \quad \omega^{(N-1) \cdot 1} \quad \cdots \quad \omega^{(N-1) \cdot (N-1)}
   \end{bmatrix}_{\underset{DFT}{}} 
   \times
   \begin{bmatrix}
   x_0 \\
   x_1 \\
   \vdots \\
   x_{N-1}
   \end{bmatrix}_{\underset{SIGNAL}{}} \\
   \text{where}\quad\omega = \text{e}^{-\frac{2\pi i}{N}}
   $$
   
3. 上述1和2是等价的,其中X_0是信号均值的N倍，也称直流分量

# FFT

1. 换一个矩阵乘法下的视角，DFT其实可以理解为求以下多项式的值

​	
$$
X_k = x_0 x^0 + x_1 x^1 + x_2 x^2 + \dots + x_{N-1} x^{N-1} \quad where \quad k = 0, 1, \dots, N-1 \quad and \quad x = \omega^k
$$

2. 假定N为2的幂次，利用如下性质，我们只需要计算w^0-w^(N/2-1)这些点的多项式的值，剩下的一半可以用对称性得到

   ![image-20240327174148691](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327174148691.png)

3. w的平方后求PePo依然满足正负对条件，重复利用2的性质再次缩减一半的计算量，以此方式递归至1项的时候直接返回

# IFFT

1. 先看计算形式
   $$
   \begin{bmatrix}
   x_0 \\
   x_1 \\
   \vdots \\
   x_{N-1}
   \end{bmatrix}_{\underset{SIGNAL}{}}
   = 
   \begin{bmatrix}
   \omega^{0 \cdot 0} \quad \omega^{0 \cdot 1} \quad \cdots \quad \omega^{0 \cdot (N-1)} \\
   \omega^{1 \cdot 0} \quad \omega^{1 \cdot 1} \quad \cdots \quad \omega^{1 \cdot (N-1)} \\
   \vdots \quad \vdots \quad \ddots \quad \vdots \\
   \omega^{(N-1) \cdot 0} \quad \omega^{(N-1) \cdot 1} \quad \cdots \quad \omega^{(N-1) \cdot (N-1)}
   \end{bmatrix}_{\underset{DFT^{-1}}{}} 
   \times
   \begin{bmatrix}
   X_0 \\
   X_1 \\
   \vdots \\
   X_{N-1}
   \end{bmatrix}_{\underset{RESULT}{}}\\
   \text{where}\quad\omega = \frac{1}{N}\text{e}^{\frac{2\pi i}{N}}
   $$

2. 我们发现与FFT的计算形式几乎一模一样，唯一的区别就是w，那么只需要将FFT中的w换成1/N倍的exp(2πi/N)即可

3. 也就是说原始信号可以由傅里叶基线性表示
   $$
   \begin{bmatrix}
   x_0 \\
   x_1 \\
   \vdots \\
   x_{N-1}
   \end{bmatrix}_{\underset{SIGNAL}{}}
   = 
   \frac{1}{N} ( X_0 B_0 + X_1 B_1 + \dots + X_{N-1} B_{N-1} ) \\
   \quad\text{where} \quad B_k = \text{e}^{\frac{2\pi i}{N}k}, \quad k = 0, 1,\dots,N-1
   $$

# FFT AND IFFT

![image-20240327180833020](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327180833020.png)

> 注意：w的幂次在FFT可以为正也可以为负，只要IFFT与之相反即可，因为正负号的区别只是做傅里叶变换的时候正着绕和反着绕的区别，FFT函数得到的两种结果区别只是互为共轭复数，通常FFT的w我们幂次取负的，也就是逆时针绕

# 离散卷积

![image-20240327191829030](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327191829030.png)

![image-20240327191608045](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327191608045.png)



1. 卷积的另一个视角其实就是**多项式乘法求系数**或者**两个向量做张量积沿对角线求和**

   ![image-20240327182042534](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327182042534.png)

   

2. 那么当卷积规模较大时候就会变成这样

   ![image-20240327182221812](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327182221812.png)

3. 有没有办法优化计算速度呢，我们引入FFT，我们知道FFT是可以求多项式值得，这里卷积就是求两个多项式的值，然后把结果乘起来，然后得到新的多项式的系数就是卷积的解
4. 如果输入m维和n维列向量，得到的结果是m+n-1维列向量，也就是说我们需要m+n-1个不同的点代入方程求m+n-1个系数，也就是多项式回归，那就是傅里叶逆变换
5. 我们假设m+n-1通常不是2的幂次，那么你需要2^ceil(log2(m+n-1))个点
6. 具体怎么做呢，假设现在有两个离散函数f(x) g(x)要做卷积，分别是m维和n维列向量，那么需要先在这两个函数末尾补0至2^ceil(log2(m+n-1))这个长度，代入2^ceil(log2(m+n-1))个点，而这些点比较特殊，正好是fft需要用的下面这些点

​	
$$
\omega_k = \text{e}^{\frac{2\pi i}{N}k}, \quad k = 0, 1,\dots,2^{\lceil log_2(m+n-1) \rceil}
$$
​	![image-20240327191207343](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327191207343.png)						

7. 那么f(x)和g(x)现在都是2^ceil(log2(m+n-1))维向量了，假设2^ceil(log2(m+n-1))就是8，接下来事情就简单了

   ![image-20240327185251313](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327185251313.png)

8. 上图我们会发现其实h(x)我们是知道的，就是代入8个点到f(x) g(x)这个两个多项式里面去求他们两个的8对解，每一对解两两相乘，我们称这种向量操作为pointwise，逐项点乘，我们得到了8个h(x)，分别对应之前求f(x) g(x)的8个点，把已知量代入上面的式子，我们发现要求的卷积解就是c0至c7，准确的说是c0至c(m+n-1)，多余的我们可以丢掉。

9. 那么怎么求c0至c7呢，我们FFT是求多项式值的，而现在是一直多项式的解，求多项式系数，那么不就是给定傅里叶变换后的值求原信号吗，那不就是IFFT吗，那么至此我们得到了用FFT加速计算的卷积的解，复杂度是O(m+n)log(m+n)

# 连续卷积

![image-20240327191458067](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327191458067.png)

1. 和离散卷积一样，两个函数想象成向量做张量积，形成的三维空间的曲面，如果沿45度对角线所在的所有曲面截面求面积，也就是沿对角线积分，得到的值就是卷积的解，准确的是卷积的解的根号2倍。

   ![image-20240327192543703](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327192543703.png)

2. 由中心极限定理我们知道服从IID的随机变量的均值是服从正态分布的，也就是说如果你对服从IID的随机变量，逐个累加(卷积)，当卷积足够多的变量后，得到的解与正态分布几乎一致，只是差了一个缩放系数

   ![image-20240327193238284](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327193238284.png)

# N不是2的幂次怎么办

1. 通常情况下傅里叶变换的信号输入都不是2的幂次，怎么解决呢，下面介绍Bluestein‘s FFT解决问题
   $$
   \begin{eqnarray*}
   X(k)
   &=&
   \sum_{n=0}^{N-1}x(n)W^{-kn}W^{\frac{1}{2}(n^2+k^2)}W^{-\frac{1}{2}(n^2+k^2)}\\
   &=& W^{-\frac{1}{2}k^2}
   \sum_{n=0}^{N-1} \left[x(n)W^{-\frac{1}{2}n^2}\right]W^{\frac{1}{2}(k-n)^2} \\
   &=& W^{-\frac{1}{2}k^2} (x_q \ast w_q)_k,\\
   where\\
   x_q(n) & = & x(n)W^{-\frac{1}{2}n^2}, \quad n=0,1,2,\ldots,N-1\\
   w_q(n) & = & W^{\frac{1}{2}n^2}, \quad n=-N+1,-N+2,\ldots,-1,0,1,2,\ldots, N-1,
   \end{eqnarray*}
   $$

2. 你可能注意到了，这里的wq他的n取值范围包括-N+1至-1这一段，正常卷积运算的这一段通常都是取0，那也就是说你计算xq与wq的卷积是得不到想要的解的

3. 不理解的可以用python做一个下图的卷积运算动画，看看如果-N+1至-1这一段不得0算出来的是什么，没错，其实就是算的wq(n)，n从-N+1至N-1这段序列与xq(n)的卷积，而且是这段卷积序列里面从N-1至2N-1的N个解

   ![image-20240327191608045](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327191608045.png)

4. 我们知道这个N可以非2的n次幂，因为之前在讲离散卷积使用FFT加速的时候允许输入的卷积序列为非2的N次幂，至此我们解决了这个问题

# 二维傅里叶变换

$$
F(u, v) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f(m, n) \cdot e^{-2\pi i \left(\frac{um}{M} + \frac{vn}{N}\right)}
$$

1. 上式里的傅里叶基B_k,l提取出来如下变换

   ![image-20240327201122521](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327201122521.png)

2. 我们会发现二维傅里叶变换这个式子本身就是与不同频率的3维傅里叶基曲面(两个不同频率的傅里叶基做张量积形成的三维曲面)做内积求他们的相似度，与一维傅里叶变换是一样的原理，这个三维的傅里叶基曲面投影到二维平面大概长下面这个样子

   ![image-20240327200933362](https://raw.githubusercontent.com/17314796423/FourierTransform/main/image-20240327200933362.png)

3. 原式进一步变换后发现也等价于对F(u, v)曲面横着每一行来一次一维傅里叶变换，再对每一列来一次一维傅里叶变换，具体过程我就不推了，能力有限，感兴趣同学自己研究一下，不过讲到这里想要实现二维傅里叶变换，代码就已经相当简单了，核心部分直接调一维的BluesteinFFT即可

### 参考

https://www.youtube.com/watch?v=h7apO7q16V0

https://youtu.be/spUNpyF58BY

https://youtu.be/KuXjwB4LzSA

https://www.youtube.com/watch?v=nmgFG7PUHfo

https://youtu.be/IaSGqQa5O-M

https://ccrma.stanford.edu/~jos/st/Bluestein_s_FFT_Algorithm.html