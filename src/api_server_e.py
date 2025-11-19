import multiprocessing as mp
import json
import time
import requests
from flask import Flask, request, jsonify
import threading

class APIServerE:
    def __init__(self, config_path="configs/config.json"):
        """
        初始化APIServerE模块
        边端API服务器, 提供API接口接收用户请求
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.host = self.config["api_server_e"]["host"]
        self.port = self.config["api_server_e"]["port"]
        self.max_connections = self.config["api_server_e"]["max_connections"]
        
        # 获取云端APIServerC的配置
        self.cloud_host = self.config["api_server_c"]["host"]
        self.cloud_port = self.config["api_server_c"]["port"]
        self.cloud_url = f"http://{self.cloud_host}:{self.cloud_port}"
        
        self.query_vectorizer = None
        self.memory_manager = None
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        self.running = False
        
    def set_query_vectorizer(self, query_vectorizer):
        """设置查询向量化器"""
        self.query_vectorizer = query_vectorizer
    
    def set_memory_manager(self, memory_manager):
        """设置记忆管理器"""
        self.memory_manager = memory_manager
    
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/query', methods=['POST'])
        def handle_query():
            """处理用户查询"""
            try:
                data = request.json
                query_text = data.get('query', '')
                
                if not query_text:
                    return jsonify({'error': '查询文本不能为空'}), 400
                
                # 生成查询ID
                query_id = int(time.time() * 1000)
                
                # 将查询添加到QueryVectorizer
                if self.query_vectorizer:
                    self.query_vectorizer.add_query(query_text, query_id)
                else:
                    return jsonify({'error': '查询向量化器未设置'}), 500
                
                # 等待向量化完成并获取结果
                vector = self._wait_for_vector(query_id)
                if vector is None:
                    return jsonify({'error': '查询向量化失败'}), 500
                
                # 将查询向量发送到MemoryManager进行检索
                if self.memory_manager:
                    query_data = {
                        'vector': vector,
                        'query_id': query_id
                    }
                    self.memory_manager.query_queue.put(query_data)
                else:
                    return jsonify({'error': '记忆管理器未设置'}), 500
                
                # 等待检索结果
                memory_results = self._wait_for_memory_results(query_id)
                if memory_results is None:
                    return jsonify({'error': '记忆检索失败'}), 500
                
                # 将查询文本和记忆结果发送到云端
                cloud_response = self._send_to_cloud(query_text, memory_results)
                
                return jsonify({
                    'query_id': query_id,
                    'status': 'processing',
                    'memory_results': len(memory_results),
                    'cloud_response': cloud_response
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取系统状态"""
            return jsonify({
                'status': 'running',
                'query_vectorizer': self.query_vectorizer is not None,
                'memory_manager': self.memory_manager is not None
            })
    
    def _wait_for_vector(self, query_id, timeout=10):
        """等待查询向量化完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.query_vectorizer and not self.query_vectorizer.vector_queue.empty():
                try:
                    vector_data = self.query_vectorizer.vector_queue.get(timeout=0.1)
                    if vector_data['query_id'] == query_id:
                        return vector_data['vector']
                    # 如果不是我们要找的查询，放回队列
                    self.query_vectorizer.vector_queue.put(vector_data)
                except:
                    pass
            time.sleep(0.1)
        return None
    
    def _wait_for_memory_results(self, query_id, timeout=10):
        """等待记忆检索结果"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.memory_manager and not self.memory_manager.result_queue.empty():
                try:
                    result_data = self.memory_manager.result_queue.get(timeout=0.1)
                    if result_data['query_id'] == query_id:
                        return result_data['results']
                    # 如果不是我们要找的结果，放回队列
                    self.memory_manager.result_queue.put(result_data)
                except:
                    pass
            time.sleep(0.1)
        return None
    
    def _send_to_cloud(self, query_text, memory_results):
        """发送查询和记忆到云端"""
        try:
            payload = {
                'query': query_text,
                'memory_results': memory_results,
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.cloud_url}/process_query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'云端请求失败: {response.status_code}'}
                
        except Exception as e:
            return {'error': f'云端通信失败: {str(e)}'}
    
    def start(self):
        """启动API服务器"""
        self.running = True
        
        # 在单独的线程中启动Flask应用
        self.flask_thread = threading.Thread(
            target=self.app.run,
            kwargs={'host': self.host, 'port': self.port, 'threaded': True}
        )
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        print(f"APIServerE已启动，地址: http://{self.host}:{self.port}")
    
    def stop(self):
        """停止API服务器"""
        self.running = False
        print("APIServerE已停止")

if __name__ == "__main__":
    # 测试代码
    api_server = APIServerE()
    try:
        api_server.start()
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        api_server.stop()