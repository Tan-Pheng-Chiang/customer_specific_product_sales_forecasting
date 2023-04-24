"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.forecasting.var import VAR

try:
    registry = MLRegistry() # create ML registry
    # VAR
    var = VAR()
    # add to ML registry
    registry.add_algorithm(endpoint_name="forecasting",
                            algorithm_object=var,
                            algorithm_name="Vector Autoregression",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Tan Pheng Chiang",
                            algorithm_description="VAR to forecast specific customer's future product sales quantity",
                            algorithm_code=inspect.getsource(VAR))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))