from flask import Flask,jsonify,request,Response
import productVectors
import json

app = Flask(__name__)


@app.route('/itarget/getrecommendation',methods=['POST'])
def get_recommendations():
	user_id = request.args.get('user_id')
	#return jsonify(productVectors.get_recommendations(user_id))
	return json.dumps(productVectors.get_recommendations(user_id)),
200,{'Content-Type':'application/json'}


							


app.run()