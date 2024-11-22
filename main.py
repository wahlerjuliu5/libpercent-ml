from fastapi import FastAPI
from dotenv import load_dotenv
from routers.train import trainRouter
from routers.predict import predictRouter
from utils import CustomORJSONResponse


load_dotenv()


# Initialize the FastAPI app
app = FastAPI()


app.get("/", response_class=CustomORJSONResponse)(lambda: {"message": "Welcome to the libpercent API!"})
#import routers
app.include_router(trainRouter)

app.include_router(predictRouter)









