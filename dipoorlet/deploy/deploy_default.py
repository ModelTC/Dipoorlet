from ..utils import dispatch_functool, logger


@dispatch_functool
def deploy_dispatcher(*args, **kwargs):
    logger.warning("Deploy Platform Not Found!")
