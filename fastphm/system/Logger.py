import logging
from colorama import Fore, Style, init


class Logger(logging.Formatter):
    """ 自定义日志格式化器，支持彩色输出 """
    __instance = None

    # __level = logging.INFO
    __level = logging.DEBUG

    __log_format = '%(levelname)s - %(asctime)s >> %(message)s'

    __COLORS = {
        'DEBUG': Fore.BLUE,        # 蓝色
        'INFO': Fore.GREEN,        # 绿色
        'WARNING': Fore.YELLOW,    # 黄色
        'ERROR': Fore.RED,         # 红色
        'CRITICAL': Fore.MAGENTA,  # 紫色
        'RESET': Style.RESET_ALL   # 重置颜色
    }

    __banner_framework = """
    ██████╗ ██╗  ██╗███╗   ███╗    ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
    ██╔══██╗██║  ██║████╗ ████║    ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
    ██████╔╝███████║██╔████╔██║    █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
    ██╔═══╝ ██╔══██║██║╚██╔╝██║    ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
    ██║     ██║  ██║██║ ╚═╝ ██║    ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
    ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    """

    __banner_author = """
     █████╗ ███╗   ██╗██████╗ ██████╗ ███████╗██╗    ██╗    ███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗ 
     ██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██║    ██║    ██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗
     ███████║██╔██╗ ██║██║  ██║██████╔╝█████╗  ██║ █╗ ██║    ███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║
     ██╔══██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██║███╗██║    ╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║
     ██║  ██║██║ ╚████║██████╔╝██║  ██║███████╗╚███╔███╔╝    ███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝
     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝     ╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝                                                                                                  
    """

    __banner_author_chinese = """
      #        #      #############                        #  #               #          
      #  ###   #                #                 #        #  #                #         
      ###  # #####      ######  #      #############       #  #     #    ##############  
     #     #   # #      #    #  #            #            #   ########   #            #  
    #     # #######     #    #  #            #            #  # #        #          # #   
     #### #    # #      ######  #            #           ##  # #          ###########    
      #  #   #####              # #          #          # # #  #  #          #           
      #  ###   #     ###############         #         #  ##   #####        #     #      
    #####  # #####              #            #            #    #           #########     
      #    #   #        ######  #            #            #    #               #   #     
      #  # #   # #      #    #  #            #            #    #   #           #         
      #   # #######     #    #  #            #            #    ######      #########     
      # # #    #        ######  #            #     #      #    #               #         
      ## # #   #        #       #     ###############     #    #               #    #    
      # #   #  # ##           # #                         #    #         #############   
             #####             #                          #    #                         
    """

    def __init__(self, log_format=None):
        # 初始化 colorama
        init(autoreset=True)

        if log_format is None:
            log_format = self.__log_format
        super().__init__(log_format, datefmt='%H:%M:%S')
        self.__logger = None

        # 创建 logger
        logger = logging.getLogger(__name__)
        logger.setLevel(self.__level)

        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 设置自定义的彩色格式化器
        ch.setFormatter(self)

        # 添加处理器到 logger
        logger.addHandler(ch)

        self.__logger = logger
        Logger.__instance = self

    def format(self, record):
        color = self.__COLORS.get(record.levelname, self.__COLORS['RESET'])
        log_fmt = f"{color}{self._style._fmt}{self.__COLORS['RESET']}"
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls()
        return cls.__instance

    @classmethod
    def debug(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.debug(string)

    @classmethod
    def info(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.info(string)

    @classmethod
    def warning(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.warning(string)

    @classmethod
    def error(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.error(string)

    @classmethod
    def critical(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.critical(string)
