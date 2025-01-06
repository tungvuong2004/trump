from http.server import HTTPServer, BaseHTTPRequestHandler
import os

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_file('index.html', 200)
        else:
            self.send_error(404)

    def serve_file(self, filename, status_code):
        try:
            file_path = os.path.join(os.getcwd(), filename)
            with open(file_path, 'rb') as file:
                content = file.read()

            self.send_response(status_code)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()

            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(500)

if __name__ == "__main__":
    host = 'localhost'
    port = 8000
    server_address = (host, port)

    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Serving HTTP on {host}:{port}...")
    httpd.serve_forever()
