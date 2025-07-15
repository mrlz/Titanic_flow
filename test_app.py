from fastapi.testclient import TestClient

from app import app
import json

def test_read_main():
    with TestClient(app) as client:
        #Dies
        Jack_Dawson = {'PassengerId': 1,
                    'Pclass': 3,
                    'Name': 'Dawson, Mr. Jack',
                    'Sex': 'male',	
                    'Age': '23',	
                    'SibSp': '0',	
                    'Parch': '0',	
                    'Ticket': '',
                    'Fare': '0',
                    'Cabin': '',
                    'Embarked': 'S'
                }



        #Lives
        Rose_DeWiit_Bukater = {
                    'Pclass': 1,
                    'Name': 'DeWiit Bukater, Miss. Rose',
                    'Sex': 'female',	
                    'Age': '17',	
                    'SibSp': '0',	
                    'Parch': '1',	
                    'Ticket': '',
                    'Fare': '53.1000',
                    'Cabin': '',
                    'Embarked': 'S'
                }




        #Lives
        Emily_Borie = {'PassengerId': '916',
                    'Pclass': 1,
                    'Name': 'Ryerson, Mrs. Arthur Larned (Emily Maria Borie)',
                    'Sex': 'female',	
                    'Age': '48',	
                    'SibSp': '1',	
                    'Parch': '3',	
                    'Ticket': 'PC 17608',
                    'Fare': '262.375',
                    'Cabin': 'B57 B59 B63 B66',
                    'Embarked': 'C'
                }


        #Dies
        Elizabeth_Watson = {'PassengerId': '925',
                    'Pclass': 3,
                    'Name': 'Johnston, Mrs. Andrew G (Elizabeth Lily" Watson)"',
                    'Sex': 'female',	
                    'Age': None,	
                    'SibSp': '1',	
                    'Parch': '2',	
                    'Ticket': 'W./C. 6607',
                    'Fare': '23.45',
                    'Cabin': '',
                    'Embarked': 'S'
                }
        #Dies
        Thomas_Henry = {'PassengerId': '950',
                    'Pclass': '3',
                    'Name': 'Davison, Mr. Thomas Henry',
                    'Sex': 'male',	
                    'Age': None,	
                    'SibSp': '1',	
                    'Parch': '0',	
                    'Ticket': '386525',
                    'Fare': '16.1',
                    'Cabin': '',
                    'Embarked': 'S'
                }


        print("Testing")
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"health_check": "OK", "model_version": 1}

        payload = [Jack_Dawson]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [0]

        payload = [Rose_DeWiit_Bukater]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [1]

        payload = [Emily_Borie]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [1]

        payload = [Elizabeth_Watson]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [0]

        payload = [Thomas_Henry]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [0]

        #Dies
        Thomas_Henry = {
                    'Pclass': '3',
                    'Name': 'Davison, Mr. Thomas Henry',
                    'Sex': 'male',	
                    'Age': None,	
                    'SibSp': '1',	
                    'Parch': '0',	
                    'Ticket': '386525',
                    'Fare': '16.1',
                    'Cabin': '',
                    'Embarked': 'S'
                }

        payload = [Thomas_Henry]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [0]

        #Dies
        Thomas_Henry = {
                    'Pclass': '5',
                    'Name': 'Davison, Mr. Thomas Henry',
                    'Sex': 'male',	
                    'Age': None,	
                    'SibSp': '1',	
                    'Parch': '0',	
                    'Ticket': '386525',
                    'Fare': '16.1',
                    'Cabin': '',
                    'Embarked': 'S'
                }

        payload = [Thomas_Henry]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == {'detail': [{'type': 'less_than', 'loc': ['body', 0, 'Pclass'], 'msg': 'Input should be less than 4', 'input': '5', 'ctx': {'lt': 4}}]}

        #Dies
        Thomas_Henry = {
                    'Pclass': '3',
                    'Name': 'Davison, Mr. Thomas Henry',
                    'Sex': 'male',	
                    'Age': None,	
                    'SibSp': '1',	
                    'Parch': '0',	
                    'Ticket': '386525',
                    'Fare': '16.1',
                    'Cabin': '',
                    'Embarked': 'S'
                }

        payload = [Jack_Dawson, Rose_DeWiit_Bukater, Emily_Borie, Elizabeth_Watson, Thomas_Henry]
        response = client.post("/predict", data=json.dumps(payload))
        assert response.json() == [0, 1, 1, 0, 0]
   


if __name__ == "__main__":
    test_read_main()