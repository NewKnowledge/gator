test:
	@echo "Building Image"
	docker-compose build
	@echo "Running Tests"
	docker-compose run gator \
		pytest --color=yes -s tests/

# notebook:
# 	@echo "Building Image"
# 	docker-compose build
# 	@echo "Starting a Jupyter notebook"
# 	docker-compose run \
# 		-p 8888:8888 \
# 		ds-service \
# 		jupyter notebook --allow-root --ip=0.0.0.0

# api:
# 	@echo "Building Image"
# 	docker-compose build
# 	@echo "Starting a web API at http://localhost:5000"
# 	docker-compose run \
# 		-p 5000:5000 \
# 		-e FLASK_APP=http_app/app.py \
# 		-e FLASK_ENV=development \
# 		ds-service flask run --host=0.0.0.0

# clean:
# 	@echo "Removing generated data and image files"
# 	rm -rf **/data/**/*.npz
# 	rm -rf **/data/**/*.pkl
# 	rm -rf **/plots/**/*.svg
# 	rm -rf **/plots/**/*.png
# 	rm -rf **/plots/**/*.jpg
