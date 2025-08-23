from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass()
class LicenseResult:
    # https://.../article/view/xxx/yyy
    url: str

    # https://creativecommons.org URLs
    licenses: list[str]

    def is_acceptable(self) -> bool:
        versions = ["4.0"]
        variants = ["by", "by-sa", "by-nc", "by-nc-sa"]
        return any(
            f"{variant}/{version}" in license
            for license in self.licenses for variant in variants for version in versions
        )

def _cc_ident(url: str) -> str:
    return urlparse(url).path.removesuffix('/')

def cc_right_in_licenses(right: str, licenses: list[str]) -> bool:
    ident = _cc_ident(right)
    return any(_cc_ident(license) == ident for license in licenses)