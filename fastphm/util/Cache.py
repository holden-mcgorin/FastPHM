import os
import pickle
import hashlib

from fastphm.system.Logger import Logger


class Cache:
    # 最大缓存文件数
    __MAX_CACHE = 3
    __CACHE_DIR = '.\\cache'

    if not os.path.exists(__CACHE_DIR):
        os.makedirs(__CACHE_DIR)

    def __init__(self, cache_dir: str):
        Cache.__CACHE_DIR = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    @classmethod
    def __get_cache_file(cls, name: str) -> str:
        """
        根据信息获取缓存文件（名称及位置）
        :param name:
        :return:
        """
        # hash_input = str(kwargs).encode('utf-8')
        # hash_value = hashlib.md5(hash_input).hexdigest()
        return os.path.join(cls.__CACHE_DIR, f'{name}.pkl')

    @classmethod
    def save(cls, target, name):
        """
        保存缓存到文件
        :return:
        """
        cache_file = cls.__get_cache_file(name)
        with open(cache_file, 'wb') as f:
            Logger.debug(f"Start generating cache file: {cache_file}")
            pickle.dump(target, f)
        Logger.debug(f"Successfully generated cache file: {cache_file}")

    @classmethod
    def load(cls, name, is_able=True):
        """
        从文件加载缓存
        :return:
        """
        if not is_able:
            return None

        cache_file = cls.__get_cache_file(name)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                Logger.debug(f"Start loading cache file: {cache_file}")
                cache = pickle.load(f)
                Logger.debug(f"Successfully loaded cache file: {cache_file}")
                return cache
        else:
            Logger.warning(f'cache file {cache_file} does not exist!')
            return None
