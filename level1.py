import numpy as np
from cable import Cable  # 导入课程提供的Cable类（保持原样）


# ------------------------------
# 1. 数据与比特流转换函数
# ------------------------------
def string_to_bits(message: str) -> list[int]:
    """将字符串转换为比特流（整数列表，0/1）"""
    bits = []
    for char in message:
        # 每个字符转换为8位二进制（ASCII编码），不足8位补前导0
        char_bits = [int(bit) for bit in f"{ord(char):08b}"]
        bits.extend(char_bits)
    return bits


def bits_to_string(bits: list[int]) -> str:
    """将比特流转换回字符串（处理可能的不完整字节）"""
    message = ""
    # 按8位一组处理
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:  # 忽略不完整的字节
            break
        # 转换为整数再转字符
        byte = int(''.join(map(str, byte_bits)), 2)
        message += chr(byte)
    return message


# ------------------------------
# 2. 调制与解调函数
# ------------------------------
def modulate(bits: list[int], samples_per_bit: int = 10) -> np.ndarray:
    """
    将比特流调制为模拟信号
    采用ASK调制：高电平（1.0）表示1，低电平（0.0）表示0
    Args:
        bits: 输入比特流（0/1列表）
        samples_per_bit: 每个比特的采样点数（提高抗噪声能力）
    Returns:
        模拟信号（numpy数组）
    """
    signal = []
    for bit in bits:
        # 每个比特用多个采样点表示，增强抗噪声能力
        signal.extend([1.0 if bit == 1 else 0.0] * samples_per_bit)
    return np.array(signal, dtype=np.float32)


def demodulate(signal: np.ndarray, samples_per_bit: int = 10) -> list[int]:
    """
    从模拟信号中恢复比特流
    Args:
        signal: 接收的模拟信号
        samples_per_bit: 每个比特的采样点数（需与调制时一致）
    Returns:
        恢复的比特流（0/1列表）
    """
    bits = []
    # 按每个比特的采样点数分组
    num_bits = len(signal) // samples_per_bit
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        # 取该组采样的平均值作为判断依据
        avg = np.mean(signal[start:end])
        # 阈值判断（0.5为高低电平的分界）
        bits.append(1 if avg > 0.5 else 0)
    return bits


# ------------------------------
# 3. 错误检测机制（校验和）
# ------------------------------
def calculate_checksum(bits: list[int]) -> list[int]:
    """
    计算简单校验和（8位）：将比特流按字节分组求和，取低8位
    Args:
        bits: 原始比特流
    Returns:
        8位校验和比特
    """
    byte_sum = 0
    # 按8位一组计算和
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            byte_bits += [0] * (8 - len(byte_bits))  # 补0
        byte = int(''.join(map(str, byte_bits)), 2)
        byte_sum += byte
    # 取低8位作为校验和
    checksum = byte_sum & 0xFF  # 确保在0-255范围内
    return [int(bit) for bit in f"{checksum:08b}"]  # 转换为8位比特


def verify_checksum(received_bits: list[int]) -> tuple[list[int], bool]:
    """
    验证校验和并提取原始比特流
    Args:
        received_bits: 包含校验和的接收比特流（原始比特+8位校验和）
    Returns:
        (原始比特流, 校验结果True/False)
    """
    if len(received_bits) < 8:
        return [], False  # 长度不足，校验失败
    # 分离原始比特和校验和
    original_bits = received_bits[:-8]
    received_checksum_bits = received_bits[-8:]
    # 计算原始比特的校验和
    calculated_checksum_bits = calculate_checksum(original_bits)
    # 比较校验和
    is_valid = (received_checksum_bits == calculated_checksum_bits)
    return original_bits, is_valid


# ------------------------------
# 4. 完整通信流程
# ------------------------------
def send_message(message: str, cable: Cable, samples_per_bit: int = 10) -> None:
    """发送端流程：数据→比特流→添加校验和→调制→传输"""
    # 1. 字符串转比特流
    original_bits = string_to_bits(message)
    # 2. 计算并添加校验和
    checksum_bits = calculate_checksum(original_bits)
    bits_with_checksum = original_bits + checksum_bits
    # 3. 调制为模拟信号
    signal = modulate(bits_with_checksum, samples_per_bit)
    # 4. 通过电缆传输（使用Cable的默认行为）
    cable.transmit(signal)
    print(f"发送成功 | 原始消息: {message} | 总比特数: {len(bits_with_checksum)}")


def receive_message(cable: Cable, samples_per_bit: int = 10) -> tuple[str, bool]:
    """接收端流程：接收信号→解调→校验和验证→恢复数据"""
    if cable.last_output_signal is None:
        return "", False  # 无接收信号
    # 1. 从电缆获取接收的模拟信号
    received_signal = cable.last_output_signal
    # 2. 解调为比特流
    received_bits = demodulate(received_signal, samples_per_bit)
    # 3. 验证校验和并提取原始比特
    original_bits, is_valid = verify_checksum(received_bits)
    # 4. 比特流转字符串
    received_message = bits_to_string(original_bits)
    return received_message, is_valid


# ------------------------------
# 测试代码（使用Cable的默认配置）
# ------------------------------
if __name__ == "__main__":
    # 直接使用Cable的默认初始化参数（不修改任何配置）
    cable = Cable()  # 完全使用默认值：length=100, attenuation=0.1, noise_level=0.01, debug_mode=False

    # 测试短消息
    short_msg = "Hello!"
    print("=== 测试短消息传输 ===")
    send_message(short_msg, cable)
    received_short, valid_short = receive_message(cable)
    print(f"接收结果 | 消息: {received_short} | 校验结果: {'成功' if valid_short else '失败'}")

    # 测试长消息
    long_msg = "这是一个更长的测试消息，用于验证点对点通信系统的稳定性。" * 5
    print("\n=== 测试长消息传输 ===")
    send_message(long_msg, cable)
    received_long, valid_long = receive_message(cable)
    print(f"接收结果 | 消息长度: {len(received_long)} | 校验结果: {'成功' if valid_long else '失败'}")

    # 输出信道统计信息
    stats = cable.get_signal_stats()
    print(f"\n信道统计 | 信噪比: {stats['snr_db']:.2f} dB | 衰减后信号幅度: {stats['output_max']:.4f}")