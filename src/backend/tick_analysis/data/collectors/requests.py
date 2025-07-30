# Stub requests module for patching in tests
class Response:
    status_code = 200
    def json(self):
        return {}

def get(*args, **kwargs):
    return Response()
