from __future__ import absolute_import

import datetime
import re

# python 2 and python 3 compatibility library
import six

import biome

PRIMITIVE_TYPES = (float, bool, bytes, six.text_type) + six.integer_types
NATIVE_TYPES_MAPPING = {
    'int': int,
    'long': int,
    'float': float,
    'str': str,
    'bool': bool,
    'date': datetime.date,
    'datetime': datetime.datetime,
    'object': object,
}


def __deserialize_primitive(data, klass):
    """Deserializes string to primitive type.

    :param data: str.
    :param klass: class literal.

    :return: int, long, float, str, bool.
    """
    try:
        return klass(data)
    except UnicodeEncodeError:
        return six.u(data)
    except TypeError:
        return data


def __deserialize_object(value):
    """Return a original value.

    :return: object.
    """
    return value


def __deserialize_date(string):
    """Deserializes string to date.

    :param string: str.
    :return: date.
    """
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string
    except ValueError:
        raise Exception(
            status=0,
            reason="Failed to parse `{0}` as date object".format(string)
        )


def __deserialize_datatime(string):
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    try:
        from dateutil.parser import parse
        return parse(string)
    except ImportError:
        return string
    except ValueError:
        raise Exception(
            status=0,
            reason=(
                "Failed to parse `{0}` as datetime object"
                    .format(string)
            )
        )


def __deserialize_model(data, klass):
    """Deserializes list or dict to model.

    :param data: dict, list.
    :param klass: class literal.
    :return: model object.
    """

    if not klass.swagger_types and not hasattr(klass,
                                               'get_real_child_model'):
        return data

    kwargs = {}
    if klass.swagger_types is not None:
        for attr, attr_type in six.iteritems(klass.swagger_types):
            if (data is not None and
                    klass.attribute_map[attr] in data and
                    isinstance(data, (list, dict))):
                value = data[klass.attribute_map[attr]]
                kwargs[attr] = to_biome_class(value, attr_type)

    instance = klass(**kwargs)

    if hasattr(instance, 'get_real_child_model'):
        klass_name = instance.get_real_child_model(data)
        if klass_name:
            instance = to_biome_class(data, klass_name)
    return instance


def to_biome_class(data, klass):
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    if type(klass) == str:
        if klass.startswith('list['):
            sub_kls = re.match('list\[(.*)\]', klass).group(1)
            return [to_biome_class(sub_data, sub_kls)
                    for sub_data in data]

        if klass.startswith('dict('):
            sub_kls = re.match('dict\(([^,]*), (.*)\)', klass).group(2)
            return {k: to_biome_class(v, sub_kls)
                    for k, v in six.iteritems(data)}

        # convert str to class
        if klass in NATIVE_TYPES_MAPPING:
            klass = NATIVE_TYPES_MAPPING[klass]
        else:
            klass = getattr(biome.spec, klass)

    if klass in PRIMITIVE_TYPES:
        return __deserialize_primitive(data, klass)
    elif klass == object:
        return __deserialize_object(data)
    elif klass == datetime.date:
        return __deserialize_date(data)
    elif klass == datetime.datetime:
        return __deserialize_datatime(data)
    else:
        return __deserialize_model(data, klass)
