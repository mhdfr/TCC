# TCC

Projeto de utilização de dados médicos de pacientes para a detecção de doenças cardíacas.
Baseado no desafio https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/

# Iniciando Ambiente

Dentro da pasta raiz execute o comando abaixo para gerar a imagem necessária:

    docker build -t jupyter-image:latest .

Assim que terminar, inicie o contêiner executando o comando abaixo. O contêiner será servido na URL localhost:8888. É possível utilizar a porta 1000 ([...] -p 1000:8888 [...]) caso a porta 8888 do host esteja ocupada.

	docker run -rm --name jupyter-work-container --hostname localhost -p 8888:8888 -v "$PWD":/home/jovyan/work jupyter-image:latest

Nota: Utilize o parâmero `--rm` para que o contêiner possa ser apagado automaticamente após ser parado.


Assim que o terminal parar de mostrar mensagens no fim deverá aparecer o seguinte texto:

	To access the notebook, open this file in a browser:
		file:///home/jovyan/.local/share/jupyter/runtime/nbserver-6-open.html
	Or copy and paste one of these URLs:
		http://localhost:8888/?token=[ACCESS_TOKEN]
		or http://127.0.0.1:8888/?token=[ACCESS_TOKEN]

Acesse a URL informada com o token de acesso no browser que a tela do Jupyter Notebook abrirá na pasta dentro do contêiner.
