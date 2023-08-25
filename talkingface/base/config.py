"""配置文件基类"""
import json
import toml
import importlib
from inspect import isclass
from dataclasses import asdict, dataclass

from talkingface.error import ComponentConfigError
from talkingface.log import logger


from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Tuple, Type

if TYPE_CHECKING:
    pass


@dataclass
class BaseConfig:
    """配置文件基类"""

    cls_components: ClassVar[List[str]] = []

    @classmethod
    def from_json(cls, path: str):
        """从Json文件中加载配置信息，初始化配置对象"""
        with open(path, "r") as f:
            data = json.load(f)
            return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str):
        """从toml文件中加载配置信息，初始化配置对象"""
        with open(path, "r") as f:
            data = toml.load(f)
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict):
        """从dict中加载配置信息，初始化配置对象"""
        data = _filter_fields(cls, data)
        return cls(**data)

    def to_dict(self) -> Dict:
        """将配置对象序列化为dict对象"""
        result = {}
        for k, v in asdict(self).items():
            if isclass(v):
                result[k] = f"{v.__module__}.{v.__name__}"
            else:
                result[k] = v
        return result

    def to_json(self, path: str):
        """将配置对象序列化到json文件中"""
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f)

    def to_toml(self, path: str):
        """将配置对象序列化到toml文件中"""
        data = self.to_dict()
        with open(path, "w") as f:
            toml.dump(data, f)

    def init_component(self, name: str) -> Any:
        """从配置文件中实例化组件，返回组件实例对象"""
        (
            component_class,
            component_config_class,
            component_config_data,
        ) = self.parse_component(name)

        return component_class(component_config_class(**component_config_data))

    def parse_component(self, name: str) -> Tuple[type, type, Dict]:
        """根据组件名解析出配置文件中的组件信息"""
        if name not in self.cls_components:
            raise ComponentConfigError(f"Invalid component {name}")

        component_class = getattr(self, f"{name}_class")
        if not component_class:
            raise ComponentConfigError(f"Missing component class: {name}")
        component_class = self._parse_class(component_class)

        component_config_class = getattr(self, f"{name}_config_class")
        if not component_config_class:
            raise ComponentConfigError(f"Missing component config class: {name}")
        component_config_class = self._parse_class(component_config_class)

        component_config_data = getattr(self, f"{name}_config_data", {})
        component_config_data = _filter_fields(
            component_config_class, component_config_data
        )

        return component_class, component_config_class, component_config_data

    @staticmethod
    def _parse_class(cls_str: str) -> "type":
        module_name, class_name = cls_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name, None)
        if not cls:
            raise ModuleNotFoundError(
                f"class {class_name} in module {module_name} not found"
            )
        return cls


def _filter_fields(cls, kwargs: Dict[str, Any]) -> Dict:
    """过滤掉dataclass中为定义的字段"""
    result = {}
    for k, v in kwargs.items():
        if k not in cls.__dataclass_fields__:
            logger.warning(f"useless field in config file: {k}={v}")
            continue

        result[k] = v
    return result


class ConfigMixin:

    config_class: "Type[BaseConfig]" = BaseConfig

    @classmethod
    def from_json(cls, path: str):
        config = cls.config_class.from_json(path)
        return cls(config)


    @classmethod
    def from_toml(cls, path: str):
        config = cls.config_class.from_toml(path)
        return cls(config)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        config = cls.config_class.from_dict(data)
        return cls(config)