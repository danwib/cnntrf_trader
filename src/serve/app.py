from fastapi import FastAPI
app=FastAPI()
@app.get('/health/ready')
def ready(): return {'status':'ok'}
