import argparse
import os
from src.api_server_e import APIServerE
from src.config import Config


def qa_loop_with_image(dialog_id: int = 1, cloud_url: str = None):
    """在指定 dialog_id 下进行「问题 + 图片」询问循环。输入空行或 'quit' 退出。"""
    config = Config()
    if cloud_url:
        config.cloud_server_url = cloud_url
    client = APIServerE(config)
    client.start()
    print(f"对话 ID={dialog_id}，依次输入问题和图片路径；空行或 quit 退出\n")
    while True:
        query_text = input("你(问题): ").strip()
        if not query_text or query_text.lower() == "quit":
            break
        image_path = input("图片路径: ").strip()
        if image_path.lower() == "quit":
            break
        if not image_path:
            image_path = None
        if image_path and not os.path.isfile(image_path):
            print(f"错误: 文件不存在 [{image_path}]\n")
            continue
        result = client.query_test(
            query_text=query_text,
            image_path=image_path,
            dialog_id=dialog_id
        )
        print(f"答: {result.get('result', '')}\n")


def single_query(query_text: str, image_path: str, dialog_id: int = 1, cloud_url: str = None):
    """单次查询：指定问题和图片路径，发送一次请求。"""
    config = Config()
    if cloud_url:
        config.cloud_server_url = cloud_url
    client = APIServerE(config)
    client.start()
    if not os.path.isfile(image_path):
        print(f"错误: 文件不存在 [{image_path}]")
        return
    result = client.query_test(
        query_text=query_text,
        image_path=image_path,
        dialog_id=dialog_id
    )
    print(f"答: {result.get('result', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="边端客户端：通过输入图片地址发送请求到云端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            使用示例:
            # 交互式循环（每次输入问题和图片路径）
            python launch_client_with_image.py

            # 指定云端地址
            python launch_client_with_image.py --url 127.0.0.1:9000

            # 单次查询（命令行指定问题和图片）
            python launch_client_with_image.py --query "图中有什么？" --image test_frame.png
        """
    )
    parser.add_argument(
        '--url',
        type=str,
        help='云端 gRPC 地址，格式 host:port (例如: 127.0.0.1:9000)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='查询内容（与 --image 一起使用时为单次查询）'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='图片文件路径（与 --query 一起使用时为单次查询）'
    )
    args = parser.parse_args()

    if args.query and args.image:
        single_query(
            query_text=args.query,
            image_path=args.image,
            cloud_url=args.url
        )
    elif args.query or args.image:
        print("错误: --query 和 --image 需同时指定才能进行单次查询")
    else:
        qa_loop_with_image(cloud_url=args.url)


if __name__ == "__main__":
    main()
