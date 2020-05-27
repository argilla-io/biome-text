#!/usr/sh

erb /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf \
&& nginx -g "daemon off;"
