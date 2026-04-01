import sys
import argparse
from src.api.api_server_e import APIServerE
from src.config import Config

def qa_loop_example(dialog_id: int = 1, cloud_url: str = None):
    """在指定 dialog_id 下进行用户询问 -> 获得回答的循环。输入空行或 'quit' 退出。"""
    config = Config()
    if cloud_url:
        config.cloud_server_url = cloud_url
    client = APIServerE(config)
    client.start()
    print(f"对话 ID={dialog_id}，输入问题后回车；空行或 quit 退出\n")
    while True:
        query_text = input("你: ").strip()
        if not query_text or query_text.lower() == "quit":
            break
        result = client.query(query_text=query_text, dialog_id=dialog_id)
        print(f"答: {result.get('result', '')}\n")

def main():
    parser = argparse.ArgumentParser(
        description="边端客户端使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            使用示例:
            # 单次查询（使用配置文件中的地址）
            python client_example.py
            
            # 指定云端地址（如通过 SSH 隧道或远程 IP）
            python test_client_connection.py --url 127.0.0.1:9000
            
            # 单次查询（指定查询内容）
            python client_example.py --query "视频中的人在做什么？"
        """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='云端 gRPC 地址，格式 host:port (例如: 127.0.0.1:9000 或 172.18.167.21:9000)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='查询内容'
    )
    
    args = parser.parse_args()
    qa_loop_example(cloud_url=args.url)


if __name__ == "__main__":
    main()