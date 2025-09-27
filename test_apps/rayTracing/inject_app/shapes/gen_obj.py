import random

def generate_random_obj(filename, num_vertices=100, num_faces=200, scale=1.0):
    """
    随机生成一个 OBJ 文件
    :param filename: 输出文件名
    :param num_vertices: 顶点数量
    :param num_faces: 三角形数量
    :param scale: 坐标缩放因子
    """
    vertices = []
    with open(filename, "w") as f:
        # 生成随机顶点
        for i in range(num_vertices):
            x = random.uniform(-1, 1) * scale
            y = random.uniform(-1, 1) * scale
            z = random.uniform(-1, 1) * scale
            vertices.append((x, y, z))
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")

        # 生成随机三角面片（注意索引 OBJ 是 1-based）
        for i in range(num_faces):
            idx = random.sample(range(1, num_vertices + 1), 3)
            f.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")

    print(f"随机 OBJ 文件已生成: {filename}")
    print(f"顶点数 = {num_vertices}, 三角形数 = {num_faces}")

# 示例：生成一个包含 50 个顶点、80 个三角形的小模型
if __name__ == "__main__":
    generate_random_obj("random_small.obj", num_vertices=10, num_faces=10, scale=5.0)
