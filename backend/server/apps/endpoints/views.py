from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import MLAlgorithm
from apps.endpoints.serializers import MLAlgorithmSerializer

from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.serializers import MLAlgorithmStatusSerializer

from apps.endpoints.models import MLRequest
from apps.endpoints.serializers import MLRequestSerializer

import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
from apps.ml.registry import MLRegistry
from server.wsgi import registry

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm = instance.parent_mlalgorithm,
                                                        created_at__lt=instance.created_at,
                                                        active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])

class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)



        except Exception as e:
            raise APIException(str(e))

class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()

class PredictView(views.APIView):
    def post(self, request, format=None):
        try:
            algs = MLAlgorithm.objects.filter(parent_endpoint__name = "forecasting", status__active=True)

            if len(algs) == 0:
                return Response(
                    {"status": "Error", "message": "ML algorithm is not available"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            alg_index = 0
            algorithm_object = registry.endpoints[algs[alg_index].id + 1]
            prediction = algorithm_object.compute_prediction(request.data)

            response = prediction["status"] if "status" in prediction else "error"
            ml_request = MLRequest(
                input_data=json.dumps(request.data),
                full_response=prediction,
                response=response,
                feedback="",
                parent_mlalgorithm=algs[alg_index],
            )
            ml_request.save()

            prediction["request_id"] = ml_request.id
            return Response({"predictions": prediction["predictions"]}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"status": "Error", "message": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
