"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import os
import io
import csv
import ssl
import uuid
import pickle
import logging
import mimetypes
try:
    import ujson as json
except:
    import json

from dateutil import parser
from rest_framework.exceptions import ValidationError
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from urllib.request import urlopen

from .models import FileUpload
from core.utils.io import url_is_local
from core.utils.exceptions import ImportFromLocalIPError

logger = logging.getLogger(__name__)
csv.field_size_limit(131072 * 10)


def is_binary(f):
    return isinstance(f, (io.RawIOBase, io.BufferedIOBase))


def csv_generate_header(file):
    """ Generate column names for headless csv file """
    file.seek(0)
    names = []
    line = file.readline()

    num_columns = len(line.split(b',' if isinstance(line, bytes) else ','))
    for i in range(num_columns):
        names.append('column' + str(i+1))
    file.seek(0)
    return names


def check_max_task_number(tasks):
    # max tasks
    if len(tasks) > settings.TASKS_MAX_NUMBER:
        raise ValidationError(f'Maximum task number is {settings.TASKS_MAX_NUMBER}, '
                              f'current task number is {len(tasks)}')


def check_file_sizes_and_number(files):
    total = sum([file.size for _, file in files.items()])

    if total >= settings.TASKS_MAX_FILE_SIZE:
        raise ValidationError(f'Maximum total size of all files is {settings.TASKS_MAX_FILE_SIZE} bytes, '
                              f'current size is {total} bytes')


def create_file_upload(request, project, file):
    instance = FileUpload(user=request.user, project=project, file=file)
    if settings.SVG_SECURITY_CLEANUP:
        content_type, encoding = mimetypes.guess_type(str(instance.file.name))
        if content_type in ['image/svg+xml']:
            clean_xml = allowlist_svg(instance.file.read())
            instance.file.seek(0)
            instance.file.write(clean_xml)
            instance.file.truncate()
    from face_detection.face_detection import face_detection_image
    # face_detection(path)
    from label_studio.core.utils.io import get_data_dir
    instance.save()

    # path = os.path.join('/home/hkuit164/.local/share/label-studio/media',instance.url[6:])
    path = os.path.join(get_data_dir(), 'media', instance.url[6:])
    face_detection_image(path)

    return instance


def allowlist_svg(dirty_xml):
    """Filter out malicious/harmful content from SVG files
    by defining allowed tags
    """
    from lxml.html import clean

    allow_tags = [
            'xml',
            'svg',
            'circle',
            'ellipse',
            'line',
            'path',
            'polygon',
            'polyline',
            'rect'
    ]

    cleaner = clean.Cleaner(
            allow_tags=allow_tags,
            style=True,
            links=True,
            add_nofollow=False,
            page_structure=True,
            safe_attrs_only=False,
            remove_unknown_tags=False)

    clean_xml = cleaner.clean_html(dirty_xml)
    return clean_xml


def str_to_json(data):
    try:
        json_acceptable_string = data.replace("'", "\"")
        return json.loads(json_acceptable_string)
    except ValueError:
        return None


def tasks_from_url(file_upload_ids, project, request, url):
    """ Download file using URL and read tasks from it
    """
    # process URL with tasks
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        filename = url.rsplit('/', 1)[-1]
        with urlopen(url, context=ctx) as file:   # nosec
            # check size
            meta = file.info()
            file.size = int(meta.get("Content-Length"))
            file.urlopen = True
            check_file_sizes_and_number({url: file})
            file_content = file.read()
            if isinstance(file_content, str):
                file_content = file_content.encode()
            file_upload = create_file_upload(request, project, SimpleUploadedFile(filename, file_content))
            file_upload_ids.append(file_upload.id)
            tasks, found_formats, data_keys = FileUpload.load_tasks_from_uploaded_files(project, file_upload_ids)

    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(str(e))
    return data_keys, found_formats, tasks, file_upload_ids


def load_tasks(request, project):
    """ Load tasks from different types of request.data / request.files
    """
    file_upload_ids, found_formats, data_keys = [], [], set()
    could_be_tasks_lists = False

    # take tasks from request FILES
    if len(request.FILES):
        check_file_sizes_and_number(request.FILES)
        for filename, file in request.FILES.items():
            file_upload = create_file_upload(request, project, file)
            if file_upload.format_could_be_tasks_list:
                could_be_tasks_lists = True
            file_upload_ids.append(file_upload.id)
        tasks, found_formats, data_keys = FileUpload.load_tasks_from_uploaded_files(project, file_upload_ids)

    # take tasks from url address
    elif 'application/x-www-form-urlencoded' in request.content_type:
        # empty url
        url = request.data.get('url')
        if not url:
            raise ValidationError('"url" is not found in request data')

        # try to load json with task or tasks from url as string
        json_data = str_to_json(url)
        if json_data:
            file_upload = create_file_upload(request, project, SimpleUploadedFile('inplace.json', url.encode()))
            file_upload_ids.append(file_upload.id)
            tasks, found_formats, data_keys = FileUpload.load_tasks_from_uploaded_files(project, file_upload_ids)
            
        # download file using url and read tasks from it
        else:
            if settings.SSRF_PROTECTION_ENABLED and url_is_local(url):
                raise ImportFromLocalIPError

            if url.strip().startswith('file://'):
                raise ValidationError('"url" is not valid')

            data_keys, found_formats, tasks, file_upload_ids = tasks_from_url(
                file_upload_ids, project, request, url
            )

    # take one task from request DATA
    elif 'application/json' in request.content_type and isinstance(request.data, dict):
        tasks = [request.data]

    # take many tasks from request DATA
    elif 'application/json' in request.content_type and isinstance(request.data, list):
        tasks = request.data

    # incorrect data source
    else:
        raise ValidationError('load_tasks: No data found in DATA or in FILES')

    # check is data root is list
    if not isinstance(tasks, list):
        raise ValidationError('load_tasks: Data root must be list')

    # empty tasks error
    if not tasks:
        raise ValidationError('load_tasks: No tasks added')

    check_max_task_number(tasks)
    return tasks, file_upload_ids, could_be_tasks_lists, found_formats, list(data_keys)

