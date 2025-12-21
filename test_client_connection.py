import sys
import argparse
from src.api_server_e import APIServerE
from src.config import Config

def send_query_example(query_text=None):
    """
    发送查询示例
    
    Args:
        cloud_url (str): 云端服务器地址
        query_text (str): 查询文本
    """
    print("=" * 60)
    print("边端客户端查询示例")
    print("=" * 60)
    
    try:
        print(f"\n从配置文件读取云端服务器地址")
        config = Config()
        client = APIServerE(config)

        print(f"云端服务器地址: {client.cloud_server_url}")
        client.start()
        # 检查连接
        print(f"\n[步骤1] 检查云端服务器连接...")
        status = client.check_cloud_status()
        
        if status:
            print(f"✓ 云端服务器运行正常")
        else:
            print(f"✗ 云端服务器状态异常")
            return False
        
        # 发送查询
        print(f"\n[步骤2] 发送查询到云端服务器...")
        
        if not query_text:
            query_text = "这是一个测试查询"
        
        print(f"  查询内容: {query_text}")
        print(f"  等待云端响应...")
        
        result = client.query(
            query_text=query_text
        )
        # result = {
        #     "query_id": grpc_response.query_id,
        #     "result": grpc_response.result,
        #     "error": grpc_response.error if grpc_response.error else None,
        #     "timestamp": grpc_response.timestamp
        # }
        print(f"\n✓ 收到云端响应")
        print(f"  - 查询ID: {result.get('query_id')}")
        print(f"  - 时间戳: {result.get('timestamp')}")
        
        if result.get('error'):
            print(f"  - 错误: {result.get('error')}")
        else:
            print(f"  - 结果: {result.get('result')}")
        
        print(f"\n" + "=" * 60)
        print(f"✓ 查询完成")
        print(f"=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 操作失败: {str(e)}")
        print(f"\n" + "=" * 60)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="边端客户端使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单次查询（使用配置文件中的地址）
  python client_example.py
  
  # 单次查询（指定云端地址）
  python client_example.py --url http://127.0.0.1:9000
  
  # 单次查询（指定查询内容）
  python client_example.py --query "视频中的人在做什么？"
  """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='云端服务器地址 (例如: http://127.0.0.1:9000)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='查询内容'
    )
    
    args = parser.parse_args()
    config = Config()
    if args.url:
        config.cloud_server_url = args.url
    success = send_query_example(args.query)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()