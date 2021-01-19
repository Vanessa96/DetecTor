import logging

logger = logging.getLogger('nrg')

logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: " \
          "%(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def sanitize(model_name):
    # todo: more robust name sanitization
    return model_name.replace('/', '_')
