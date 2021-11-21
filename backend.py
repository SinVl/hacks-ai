from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
from starlette.responses import StreamingResponse, RedirectResponse, FileResponse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tools.preprocessing import unify_sym, process, process_pipeline

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="")

def analyzer(x):
    return x
rfc = pickle.load(open('models/rfc_latest.pickle', 'rb'))
tfidf = pickle.load(open('models/tfidf_latest.pickle', 'rb'))
subcat_df = pd.read_csv('models/subcat.csv')
subcat_dict = {}
for i, row in subcat_df.iterrows():
    subcat_dict[row['common_code']] = row['subcat']

    
def predict(data):
    input_data = data.strip()
    res = unify_sym(input_data)
    output = process(process_pipeline, text=res)[:25]
    tr = tfidf.transform([output])
    ep_code = rfc.predict(tr.toarray())[0]
    subcat = subcat_dict[ep_code]
    
    return '.'.join(ep_code.split('_')), subcat    


@app.get("/")
def read_root1(request: Request):
    return RedirectResponse('/index1')


@app.get("/index1", response_class=HTMLResponse)
def read_root1(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})


@app.get("/index2", response_class=HTMLResponse)
def read_root2(request: Request):
    return templates.TemplateResponse("index2.html", {'request': request})


@app.get("/index3", response_class=HTMLResponse)
def read_root3(request: Request):
    return templates.TemplateResponse("index3.html", {'request': request})


@app.post("/index3")
async def create_file(files: UploadFile = File(...)):
    try:
        if 'xlsx' not in files.filename:
            raise ValueError
        with open(files.filename, 'wb') as xlsx:
            content = await files.read()
            xlsx.write(content)
            xlsx.close()
        df = pd.read_excel(f'{files.filename}')
        if 'Общее наименование продукции' not in df.columns or 'Раздел ЕП РФ (Код из ФГИС ФСА для подкатегории продукции)' not in df.columns:
            raise ValueError
        
        df = df.rename(columns={'Общее наименование продукции': 'product_name', 'Раздел ЕП РФ (Код из ФГИС ФСА для подкатегории продукции)': 'ep_code'})
        
        df['ep_code'] = df['ep_code'].astype(str)
        data = []
        df=df[:200]
        for i, row in df.iterrows():
            ep_code, subcat = predict(row['product_name'])
            data.append(ep_code==row['ep_code'])
        df['validity'] = data
        
        df = df.rename(columns={'product_name': 'Общее наименование продукции', 'ep_code': 'Раздел ЕП РФ (Код из ФГИС ФСА для подкатегории продукции)', 'validity': 'Валидность'})
        
        df.to_csv('validated.csv')
        file_like = open('validated.csv', mode="rb")
        return FileResponse('validated.csv', media_type='application/octet-stream', filename='validated.csv') #, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except ValueError:
        return {'error': 'Некорректный файл!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'}


@app.get("/get_chance/{product_name}/{cat1}/{cat2}")
def get_chance(product_name: str, cat1: str, cat2: str):
    client_ep_code = f'{cat1.strip()}.{cat2.strip()}'
    
    ep_code, subcat = predict(product_name)
    
    if client_ep_code != ep_code:
        return {'result': 'error'}
    else:
        return {'result': 100}


@app.get("/predict_category/{current_string}")
async def read_item(current_string: str):
    ep_code, subcat = predict(current_string)
    
    
    predict1_str = 'Медикаменты'
    predict2_str = 'Медикаменты для детей'
    #ep_code = '1111.1'
    return {"predict_1": '-',
            "predict_2": subcat,
            "ep_code": ep_code}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.108", port=8886)



