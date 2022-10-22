#!/bin/bash
celery  -A hamp_pred.app.app.celery worker -l INFO --pidfile /tmp/celery.pid