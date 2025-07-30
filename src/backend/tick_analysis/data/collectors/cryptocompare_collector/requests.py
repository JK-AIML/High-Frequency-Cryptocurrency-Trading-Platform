# Stub requests module for test patching compatibility
class Response:
    status_code = 200
    def json(self):
        return {}

def get(*args, **kwargs):
    return Response()
