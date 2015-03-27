#include "stdafx.h"
#include "FilterTIG.h"
#include "CmShow.h"
#include <iomanip>

void FilterTIG::update(CMat &w1f){
	CV_Assert(w1f.cols * w1f.rows == D && w1f.type() == CV_32F && w1f.isContinuous());
	float b[D], residuals[D];
	memcpy(residuals, w1f.data, sizeof(float)*D);
	for (int i = 0; i < NUM_COMP; i++){
		float avg = 0;
		for (int j = 0; j < D; j++){
			b[j] = residuals[j] >= 0.0f ? 1.0f : -1.0f;
			avg += residuals[j] * b[j];
		}
		avg /= D;
		_coeffs1[i] = avg, _coeffs2[i] = avg*2, _coeffs4[i] = avg*4, _coeffs8[i] = avg*8;
		for (int j = 0; j < D; j++)
			residuals[j] -= avg*b[j];
		UINT64 tig = 0;
		for (int j = 0; j < D; j++)
			tig = (tig << 1) | (b[j] > 0 ? 1 : 0);
		_bTIGs[i] = tig;
	}
}

void FilterTIG::reconstruct(Mat &w1f){
	w1f = Mat::zeros(8, 8, CV_32F);
	float *weight = (float*)w1f.data;
	for (int i = 0; i < NUM_COMP; i++){
		UINT64 tig = _bTIGs[i];
		for (int j = 0; j < D; j++)
			weight[j] += _coeffs1[i] * (((tig >> (63-j)) & 1) ? 1 : -1);
	}
}

// 论文中的算法2。传入函数的 mag1u 是完整图像的梯度图，其大小会超过8*8。而单个窗口是 8*8
// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
// Please refer to my paper for definition of the variables used in this function
// 算法解释：
// 这里面涉及到了三种“图”：梯度图、二进制的梯度图（BING）、分数图（BING与分类器做内积得到的分数）
// 这面这套算法用于计算整张梯度图的分数图。梯度图中的每个“像素”与分数图中的“像素”一一对应。这时，你可能会问。
// BING图是8*8区域的，那么为什么梯度图会与分数图一一对应呢？这是因为，分数的计算，是在梯度图上密集的区域进行的。
// 所谓密集的区域是指，区域与区域之间有交集，且交集非常大。大到什么程度呢？在本算法中，大到只有 1bit 的不同。
// 也就是说，从梯度图中的第一个“像素”开始，计算一个8*8范围内的分数，然后再移到下一个“像素”，再计算8*8区域的分数。
// 这种方法是合乎情理的，因为我们要判断一个区域是否有目标，是以像素为精度的。每一个像素都有可能在或者不在目标身上。
// 所以，每个像素都要计算分数。整体过程如下：
// 首先，将图像缩放到一定的大小，假设是 8*8像素。然后计算图像的梯度图，mag1u。显然，mag1u也是 8*8 像素，因为图像中的每个像素
// 都是平起平坐的，每个像素都应该有“梯度”这一属性。所以 mag1u 是 8*8 的。这里要注意的是，梯度的取值范围是 0-255，因此，存储单位是 byte。
// 可以把 mag1u 想象为 8*8 的矩阵，每个元素都用 byte 存储，即 8 bit。然后，根据论文的设置，这 8 bit 中，我只取前 4 Bit，后面 4 bit 不要了。
// 这样，就把 mag1u 拆分为了 4 张以 bit 为单位的二进制图。每张二进制图的存储单位都是 1 bit。接下来的工作就是，挖取二进制图中的 8*8 的区域，形成一个 BING 特征
// 然后，在二进制图上，以 bit 为单位进行滑动，分别提取每一个位置附近的 8*8 的二进制数，分别形成 BING 特征。这样，最终就构成了一系列 BING 特征。
// 把这些 BING 特征存放到以 INT64 为单位的 Mat 矩阵 Tig 中，形成 BING 图。由于有 4 张 二进制图，因此需要 4 张 BING 图与之对应，即 Tig1-8。
// 至此，可以看出，Tig 存储的是每个位置上的 BING 特征。相应的，Row 就是每个 BING 特征的最后一行。由于每个位置都有一个 BING 特征，所以每个位置也都有
// “最后一行”，因此 Row 也是 8*8 大小的。
// 最后，关于边界处理问题。当我们处理图像的中间区域时，是不存在边界问题的。按照算法进行即可。但是当位于边界时，像素周围是没有值的，是空的。
// 那么问题是，此时是否仍然可以采用相同的处理方法来处理边界像素呢？答案是可以。因为，从本质上讲，边界像素和中心像素是平起平坐的。边界的概念只是相对的
// 当我们把边界外面在围一圈像素时，边界像素就转化为了中心像素。可见，边界像素和中心像素在概念上可以互相转化，而数值上没有任何变化。他们都遵循
// 相同的规律。因此，可以直接把中心像素的处理方法应用到边界像素上面。所需的额外工作仅仅是外面围一圈像素即可。
Mat FilterTIG::matchTemplate(const Mat &mag1u){
	const int H = mag1u.rows, W = mag1u.cols; // 获取梯度图的行列数
	const Size sz(W+1, H+1); // Expand original size to avoid dealing with boundary conditions
    
    // 生成全零矩阵，INT64: long long 类型； byte：unsigned char 类型。
    // paper: First, a BING feature b and its x,y last row r could be saved in
    // a single INT64 and a BYTE x,y variables, respectively.
	Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);
	Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);
	Mat_<byte> Row1 = Mat_<byte>::zeros(sz), Row2 = Mat_<byte>::zeros(sz);
	Mat_<byte> Row4 = Mat_<byte>::zeros(sz), Row8 = Mat_<byte>::zeros(sz);
	
    // 分数图的大小和梯度图的大小一样。
    Mat_<float> scores = Mat_<float>::zeros(sz); // 访问scores，可以直接像数组一样用 scores(x,y) 即可。
    
    // 这一套循环可以计算完整个梯度图的分数。如果不采用 bitewise operation 的话，里面就还要再嵌套一个循环。
    // 这里 T1,T2,T4,T8 为什么要取 4 个指针呢？这是因为，根据论文中所说，Ng = 4。也就是对该梯度图中的每个“像素”，
    // 像素的前 4 个 bit 来做近似。因此，一个梯度图，会被分解为 4 个以 bit 为存储单位的 bit图。
    // 在做分数计算的时候，这四个 bit 图是相互独立的。即每个 bit 图都要计算分数。同时，由于这里采用了 bitwise operation
    // 所以，仅仅需要一套循环就可以同时计算 4 个 bit 图的分数。
	for(int y = 1; y <= H; y++){  // for each row
		const byte* G = mag1u.ptr<byte>(y-1); // 第一次循环的时候是处理梯度图的第 0 行。
        
        // T1-T8 代表当前正在计算的 BING 特征。注意，bit 图中每个 8*8 大小的区域就是一个 BING 特征
		INT64* T1 = Tig1.ptr<INT64>(y); // Binary TIG of current row
		INT64* T2 = Tig2.ptr<INT64>(y);
		INT64* T4 = Tig4.ptr<INT64>(y);
		INT64* T8 = Tig8.ptr<INT64>(y);
        
        // Tu1-Tu8 代表当前正在计算的 BING 特征的上侧的 BING 特征。“u” 是 up 的简写。
        // Tu1 与 T1 在上下方向上，只偏移了 1 bit。同理，T1 左侧的 BING 特征，相对于 T1 也是只偏移 1 bit
		INT64* Tu1 = Tig1.ptr<INT64>(y-1); // Binary TIG of upper row
		INT64* Tu2 = Tig2.ptr<INT64>(y-1);
		INT64* Tu4 = Tig4.ptr<INT64>(y-1);
		INT64* Tu8 = Tig8.ptr<INT64>(y-1);
        
        // R1-R8[x] 对应 bit 图中 T1-T8 的最后一行。而R1-R8[x-1]则对应 T1-T8 左侧 1 bit 位置的 BING 的最后一行
		byte* R1 = Row1.ptr<byte>(y);
		byte* R2 = Row2.ptr<byte>(y);
		byte* R4 = Row4.ptr<byte>(y);
		byte* R8 = Row8.ptr<byte>(y);
		float *s = scores.ptr<float>(y);
		for (int x = 1; x <= W; x++) { // for each column
			byte g = G[x-1]; // 首次循环的时候，处理的是梯度图的第 0 个像素，即填充的像素。
			R1[x] = (R1[x-1] << 1) | ((g >> 4) & 1); // 这里 & 1 的作用是取最后一行的最后一个元素,即论文中的 bxy。
			R2[x] = (R2[x-1] << 1) | ((g >> 5) & 1); // (g >> x) & 1 这个式子是取 g 的第 x+1 位比特。
			R4[x] = (R4[x-1] << 1) | ((g >> 6) & 1); // 所以，这四行的意思就是取 g 的第 5，6，7，8 位比特
			R8[x] = (R8[x-1] << 1) | ((g >> 7) & 1); // 即 g 的高 8 位。即，梯度图中每个“像素”的前 4 个 bit
			T1[x] = (Tu1[x] << 8) | R1[x];  // 根据算法2猜测，这里的T1就是一个BING特征图。
			T2[x] = (Tu2[x] << 8) | R2[x];  // R2 就是一个BING特征图的最后一行
			T4[x] = (Tu4[x] << 8) | R4[x];
			T8[x] = (Tu8[x] << 8) | R8[x];
			s[x] = dot(T1[x], T2[x], T4[x], T8[x]); // 论文中的公式 6。
            cout << "scores:\n" << fixed << setprecision(10) << scores << endl;
		}
	}
	Mat matchCost1f;
    // Rect(8, 8, W-7, H-7) 之外的区域，会将填充“像素”计算了进去，因此忽略掉。
    // 这时，你可能会问：不是说梯度图与分数图是一一对应的吗？如果将分数图中的一部分忽略掉，那岂不是导致了梯度图中同位置的像素就没有分数了吗？
    // 答案是：这里说的分数图与梯度图一一对应，是指梯度图中每个位置上的像素都需要计算一个分数，即梯度图中的每个位置都会得到一个分数。
    // 但是这句话并没有说，我是如何计算分数的。实际上，每个位置分数的计算，都是以该像素为基础，其左上角8*8区域内的“区域”分数。
    // 也就是说，分数是针对区域的，而不是单个像素。因此，即使我在分数图中丢弃了一部分结果，但是，剩余部分的分数，仍然包含了整个梯度图的信息。
    // 所以不会造成信息损失。Rect(8, 8, W-7, H-7) 是梯度图的整个右下角，从点（8，8）右至图像右边界，下至图像下边界。
	scores(Rect(8, 8, W-7, H-7)).copyTo(matchCost1f);
    cout << "scores:\n" << scores << endl;
    cout << "matchCost1f: \n" << matchCost1f << endl;
	return matchCost1f;
}
