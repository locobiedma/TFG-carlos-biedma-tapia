from django.shortcuts import render_to_response
from django.template import RequestContext

def mediar(media, densidad):
	operacion = (int(media) + int(densidad))/2
	return operacion


def saludar(request):
	edad = ''
	densidad = ''
	media = 0
	respuesta = ''

	if request.method == "POST":
		print "edad es: " + str(request.POST.get("edad"))
		edad = str(request.POST.get("edad"))
		print "densidad es: " + str(request.POST.get("densidad"))
		densidad = str(request.POST.get("densidad"))

		if (edad != '' and densidad != ''):
			media = mediar(edad,densidad)
			respuesta = "la media es: " + str(media)
		else:
			respuesta = "parametros incorrectos"

	elif request.method == "GET":
		print "he recibido un get"

	

	return render_to_response('plantilla.html',{'lista':'hola', 'nombre': respuesta}, context_instance=RequestContext(request))