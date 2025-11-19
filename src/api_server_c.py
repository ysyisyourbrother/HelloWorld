import multiprocessing as mp
import json
import time
from flask import Flask, request, jsonify, Response
import threading
import queue

class APIServerC:
    def __init__(self, config_path="configs/config.json"):
        """
        初始化APIServerC模块
        云端API服务器，负责接收客户端查询并返回推理结果
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.host = self.config["api_server_c"]["host"]
        self.port = self.config["api_server_c"]["port"]
        self.max_connections = self.config["api_server_c"]["max_connections"]
        
        self.reasoner = None
        self.active_streams = {}  # 存储活跃的流式连接
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        self.running = False
        
    def set_reasoner(self, reasoner):
        """设置推理器"""
        self.reasoner = reasoner
    
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/process_query', methods=['POST'])
        def process_query():
            """处理查询请求"""
            try:
                data = request.json
                query_text = data.get('query', '')
                memory_results = data.get('memory_results', [])
                stream_mode = data.get('stream_mode', True)
                
                if not query_text:
                    return jsonify({'error': '查询文本不能为空'}), 400
                
                # 生成查询ID
                query_id = int(time.time() * 1000)
                
                # 将查询添加到Reasoner
                if self.reasoner:
                    self.reasoner.add_query(
                        query_text, 
                        memory_results, 
                        query_id, 
                        stream_mode
                    )
                else:
                    return jsonify({'error': '推理器未设置'}), 500
                
                if stream_mode:
                    # 流式响应
                    return Response(
                        self._stream_response(query_id),
                        mimetype='text/plain',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Query-ID': str(query_id)
                        }
                    )
                else:
                    # 批量响应
                    result = self._wait_for_result(query_id)
                    if result:
                        return jsonify(result)
                    else:
                        return jsonify({'error': '推理失败'}), 500
                        
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取系统状态"""
            return jsonify({
                'status': 'running',
                'reasoner': self.reasoner is not None,
                'active_streams': len(self.active_streams)
            })
        
        @self.app.route('/stop_stream/<query_id>', methods=['POST'])
        def stop_stream(query_id):
            """停止流式响应"""
            if query_id in self.active_streams:
                del self.active_streams[query_id]
                return jsonify({'status': 'stopped'})
            else:
                return jsonify({'error': '流不存在'}), 404
    
    def _stream_response(self, query_id):
        """生成流式响应"""
        self.active_streams[query_id] = True
        
        try:
            while query_id in self.active_streams:
                if self.reasoner and not self.reasoner.result_queue.empty():
                    try:
                        result_data = self.reasoner.result_queue.get(timeout=0.1)
                        if result_data['query_id'] == query_id:
                            if 'chunk' in result_data:
                                chunk_data = result_data['chunk']
                                if chunk_data.get('finished', False):
                                    # 流结束
                                    del self.active_streams[query_id]
                                    yield "data: [DONE]\n\n"
                                    break
                                else:
                                    # 发送数据块
                                    chunk = chunk_data.get('chunk', '')
                                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                            elif 'result' in result_data:
                                # 批量结果
                                yield f"data: {json.dumps(result_data['result'])}\n\n"
                                del self.active_streams[query_id]
                                break
                            elif 'error' in result_data:
                                # 错误信息
                                yield f"data: {json.dumps({'error': result_data['error']})}\n\n"
                                del self.active_streams[query_id]
                                break
                        else:
                            # 不是我们要找的结果，放回队列
                            self.reasoner.result_queue.put(result_data)
                    except queue.Empty:
                        pass
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        del self.active_streams[query_id]
                        break
                
                time.sleep(0.01)  # 减少CPU占用
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if query_id in self.active_streams:
                del self.active_streams[query_id]
    
    def _wait_for_result(self, query_id, timeout=30):
        """等待批量结果"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.reasoner and not self.reasoner.result_queue.empty():
                try:
                    result_data = self.reasoner.result_queue.get(timeout=0.1)
                    if result_data['query_id'] == query_id:
                        return result_data
                    # 如果不是我们要找的结果，放回队列
                    self.reasoner.result_queue.put(result_data)
                except queue.Empty:
                    pass
            time.sleep(0.1)
        return None
    
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
        
        print(f"APIServerC已启动，地址: http://{self.host}:{self.port}")
    
    def stop(self):
        """停止API服务器"""
        self.running = False
        self.active_streams.clear()
        print("APIServerC已停止")

if __name__ == "__main__":
    # 测试代码
    api_server = APIServerC()
    try:
        api_server.start()
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        api_server.stop()