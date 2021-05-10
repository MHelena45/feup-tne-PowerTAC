from http.server import BaseHTTPRequestHandler, HTTPStatus, ThreadingHTTPServer
from ssl import wrap_socket, PROTOCOL_TLS


class OurBaseHandler(BaseHTTPRequestHandler):
    def _set_OK_response(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    
    def do_GET(self):
        self._set_OK_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        print(f"""POST request,
        Headers:
        {str(self.headers)}
        
        Body:
        {post_data.decode('utf-8')}
        """)

        self._set_OK_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def serve_endpoint(address, port):
    server_address = (address, port)

    server = ThreadingHTTPServer(server_address, OurBaseHandler)
    server.socket = wrap_socket(server.socket,
                                server_side=True,
                                certfile='server.pem',
                                ssl_version=PROTOCOL_TLS)

    server.serve_forever()

serve_endpoint('localhost', 4443)