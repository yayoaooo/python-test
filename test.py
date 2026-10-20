import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def cv_imread(file_path, flag=cv2.IMREAD_GRAYSCALE):
    file_data = np.fromfile(file_path, dtype=np.uint8)
    cv_img = cv2.imdecode(file_data, flag)
    return cv_img

def segment_characters(processed_plate):
    """
    从预处理后的车牌图像中分割单个字符
    :param processed_plate: 预处理后的二值化车牌图像
    :return: 按水平顺序排列的单个字符图像列表
    """
    print("\n=== 字符分割调试信息 ===")
    
    img_height, img_width = processed_plate.shape
    print(f"图像尺寸: {img_width}x{img_height}")
    
    # 创建工作副本
    working_img = processed_plate.copy()
    
    # 使用固定位置分割（中国车牌字符位置相对固定）
    # 车牌标准尺寸约为450x130，字符布局为：
    # 中文字符(1) + 字母(1) + 点(1) + 字母/数字(5) = 8个区域
    # 但实际识别中通常忽略点，识别7个字符
    
    # 手动定义字符位置范围（基于450x130的图像）
    # 第1个字符（中文）: 更宽
    # 第2-7个字符（字母/数字）: 宽度较均匀
    
    char_regions = [
        (15, 65),   # 中文字符（更宽）
        (85, 55),   # 字母
        (155, 50),  # 字母/数字
        (220, 50),  # 字母/数字
        (285, 50),  # 字母/数字
        (350, 50),  # 字母/数字
        (410, 35)   # 数字（可能更窄）
    ]
    
    segmented_chars = []
    
    for idx, (start_x, width) in enumerate(char_regions):
        end_x = min(start_x + width, img_width)
        
        # 提取垂直方向的有效区域
        vertical_slice = working_img[:, start_x:end_x]
        rows_with_content = np.where(np.sum(vertical_slice, axis=1) > 0)[0]
        
        if len(rows_with_content) > 0:
            y_start = max(0, rows_with_content[0] - 2)
            y_end = min(img_height, rows_with_content[-1] + 3)
            h = y_end - y_start
            
            if h > 15:
                single_char = working_img[y_start:y_end, start_x:end_x]
                
                # 根据字符类型调整缩放（第一个字符是中文，更宽）
                if idx == 0:
                    resized_char = cv2.resize(single_char, (24, 40))
                else:
                    resized_char = cv2.resize(single_char, (20, 40))
                
                _, binary_char = cv2.threshold(resized_char, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                segmented_chars.append(binary_char)
                print(f"字符{idx+1}: 位置({start_x}:{end_x}), 原始尺寸({width}x{h}), 缩放后({resized_char.shape[1]}x{resized_char.shape[0]})")
    
    print(f"\n成功分割到 {len(segmented_chars)} 个字符")
    
    if len(segmented_chars) != 7:
        print(f"警告：期望分割7个字符，实际分割到{len(segmented_chars)}个")
    
    return segmented_chars


# -------------------------- 模板加载函数 --------------------------
def load_char_templates(template_folder):
    """
    从指定文件夹加载字符模板
    :param template_folder: 模板文件夹路径
    :return: 模板列表，每个元素为(template_img, char_label)元组
    """
    templates = []
    # 遍历文件夹中的所有jpg文件
    for filename in os.listdir(template_folder):
        if filename.lower().endswith('.jpg'):
            # 提取文件名作为字符标签（不含扩展名）
            char_label = os.path.splitext(filename)[0]
            template_path = os.path.join(template_folder, filename)
            # 读取模板（灰度图）并二值化
            template_img = cv_imread(template_path)
            _, binary_template = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            templates.append((binary_template, char_label))

    if not templates:
        raise ValueError(f"错误：在{template_folder}中未找到任何jpg模板文件")
    return templates


# -------------------------- 字符识别函数 --------------------------
def recognize_characters(segmented_chars, templates, match_threshold=0.4):
    """
    基于模板匹配识别分割后的字符
    :param segmented_chars: 分割后的字符图像列表
    :param templates: 加载好的字符模板列表
    :param match_threshold: 匹配阈值（≥该值视为匹配成功，默认0.4）
    :return: 识别结果列表，匹配失败的字符用"?"表示
    """
    recognition_result = []

    for char_idx, single_char in enumerate(segmented_chars):
        best_score = -1  # 初始化最高相似度（0~1）
        best_char = "?"  # 初始化最佳匹配字符
        char_height, char_width = single_char.shape

        # 遍历所有模板进行匹配
        for (template_img, template_label) in templates:
            # 确保模板尺寸与输入字符一致
            template_resized = cv2.resize(template_img, (char_width, char_height))

            # 使用归一化相关系数匹配（适合车牌字符）
            match_result = cv2.matchTemplate(single_char, template_resized, cv2.TM_CCOEFF_NORMED)
            current_score = np.max(match_result)

            # 更新最佳匹配
            if current_score > best_score:
                best_score = current_score
                best_char = template_label

        # 根据阈值判断匹配结果
        final_char = best_char if best_score >= match_threshold else "?"
        recognition_result.append(final_char)

        # 打印单个字符匹配信息
        print(f"第{char_idx + 1}个分割字符 → 匹配结果：{final_char}（相似度：{best_score:.2f}）")

    return recognition_result

def method2_hsv_color_space(image_path):
    # ========== 步骤1：HSV颜色空间提取蓝色区域 ==========
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图像，请检查路径！")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR转HSV
    roihist=cv2.calcHist([hsv], [0], None, [256], [0, 256])

    # 定义蓝色的HSV范围（OpenCV中H:0-179, S/V:0-255）
    lower_blue = np.array([100, 150, 150])  # 最低蓝色
    upper_blue = np.array([124, 255, 255])  # 最高蓝色
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 提取蓝色区域
    #cv2.imshow("blue_mask", blue_mask)

    plt.subplot(131), plt.imshow(hsv)
    plt.subplot(132), plt.plot(roihist)
    plt.subplot(133), plt.imshow(blue_mask)
    plt.show()

    # ========== 步骤2：形态学操作去噪 ==========
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=13)  # 膨胀（填补漏洞）
    blue_mask = cv2.erode(blue_mask, kernel, iterations=13)  # 腐蚀（去除噪点）
    cv2.imshow("blue_mask_open", blue_mask)

    # ========== 步骤3：边缘检测 ==========
    edges = cv2.Canny(blue_mask, 60, 150)
    #cv2.imshow("edges", edges)

    # ========== 步骤4：轮廓提取与车牌定位 ==========
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_roi = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / h
        if 2.5 < aspect_ratio < 5.5 and w > 150 and h > 50:
            plate_roi = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    if plate_roi is None:
        print("❌ 未检测到车牌！")
        return
    cv2.imshow("plate_roi_first", plate_roi)
    #cv2.imshow("plate_roi", plate_roi)
    plate_roi=cv2.resize(plate_roi, (450,130))
    # ========== 步骤5：字符分割预处理 ==========
    #彩色图像转换为灰度图
    plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    #灰度图进行双边滤波删除细节
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    #cv2.imshow("plate_gray", plate_gray)
    #反二值化和自动阈值化处理
    _, plate_bin = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("plate_gray_bin", plate_bin)

    img_mask=np.ones(plate_bin.shape,dtype=np.uint8)
    img_mask[10:120,15:440]=255
    cv2.imshow("plate_gray_mask", img_mask)
    plate_bin=cv2.bitwise_and(plate_bin,img_mask)
    cv2.imshow("cut_bin", plate_bin)
    plate_bin=cv2.medianBlur(plate_bin, 5)

    cv2.imshow("filter_bin", plate_bin)

    '''
    #二值图像放大
    #plate_bin=cv2.resize(plate_bin, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)
    #cv2.imshow("plate_bin", plate_bin)

    kernel = np.ones((3, 3), np.uint8)
    plate_bin=cv2.dilate(plate_bin, kernel, iterations=3)
    plate_bin=cv2.erode(plate_bin, kernel, iterations=3)
    cv2.imshow("close_bin", plate_bin)
    
    
    #进行膨胀处理，将车牌字符变大
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    plate_bin=cv2.dilate(plate_bin, kernel, iterations=5)
    plate_bin=cv2.erode(plate_bin, kernel, iterations=5)
    cv2.imshow("connect_plate", plate_bin)

    #进行膨胀加腐蚀，去除黑点
    #plate_bin = cv2.dilate(plate_bin, None, iterations=3)
    #plate_bin = cv2.erode(plate_bin, None, iterations=3)
    #cv2.imshow("open", plate_bin)
    #plate_bin=cv2.bitwise_not(plate_bin)
    #plate_bin=cv2.erode(plate_bin, kernel, iterations=5)

    #图像尺寸复原
    #plate_bin = cv2.resize(plate_bin, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)

    #plate_bin=cv2.dilate(plate_bin, None, iterations=3)
    #plate_bin = cv2.resize(plate_bin, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)
    #kernel1=cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    #plate_bin=cv2.dilate(plate_bin, kernel1, iterations=3)
    #cv2.imshow("plate_bin_final", plate_bin)
    '''
    processed_plate=plate_bin
    #cv2.imwrite("D:/pics/chepai.jpg", processed_plate)

    # ========== 步骤：字符轮廓查找加模板匹配 ==========
    # 4. 字符分割
    print("步骤4：分割车牌字符...")
    segmented_chars = segment_characters(processed_plate)
    if not segmented_chars:
        return "字符分割失败"

    # 5. 加载字符模板
    print("步骤5：加载字符模板...")
    templates = load_char_templates(template_folder)

    # 6. 字符识别
    match_threshold = 0.4
    print("步骤6：识别字符...")
    recognition_result = recognize_characters(segmented_chars, templates, match_threshold)

    # 7. 生成最终结果
    final_result = "".join(recognition_result)
    print(f"\n最终识别结果：{final_result}")

    # ========== 步骤6：结果输出 ==========
    cv2.imshow('HSV Blue Mask', blue_mask)
    cv2.imshow('Original Image (Plate Marked)', img)
    #cv2.imshow('Sec_Plate_ROI', plate_roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "./resources/car333.jpg"  # 替换为你的车牌图片路径
    template_folder = "D:/pics/img/"
    print("\n=== HSV颜色空间法 ===")
    method2_hsv_color_space(image_path)