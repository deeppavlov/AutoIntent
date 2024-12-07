import abc
import re
from typing import ClassVar, Literal

import nbformat
from jupytext import jupytext
from pydantic import BaseModel


class ReplacePattern(BaseModel, abc.ABC):
    """
    An interface for replace patterns.
    """

    @property
    @abc.abstractmethod
    def pattern(self) -> re.Pattern:
        """
        A regex pattern to replace in a text.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def replacement_string(matchobj: re.Match) -> str:
        """
        Return a replacement string for a match object.

        :param matchobj: A regex match object.
        :return: A string to replace match with.
        """
        ...

    @classmethod
    def replace(cls, text: str) -> str:
        """
        Replace all instances of `pattern` in `text` with the result of `replacement_string`.

        :param text: A text in which patterns are replaced.
        :return: A string with patterns replaced.
        """
        return re.sub(cls.pattern, cls.replacement_string, text)


class InstallationCell(ReplacePattern):
    """
    Replace installation cells directives.

    Uncomment `# %pip install {}`, add a "quiet" flag, add a comment explaining the cell.
    """

    pattern: ClassVar[re.Pattern] = re.compile("\n# %pip install (.*)\n")

    @staticmethod
    def replacement_string(matchobj: re.Match) -> str:
        return f"""
# %%
# installing dependencies
%pip install -q {matchobj.group(1)}
"""


class DocumentationLink(ReplacePattern):
    """
    Replace documentation linking directives.

    Replace strings of the `%doclink({args})` format with corresponding links to local files.

    `args` is a comma-separated string of arguments to pass to the :py:meth:`.DocumentationLink.link_to_doc_page`.

    So, `%doclink(arg1,arg2,arg3)` will be replaced with `link_to_doc_page(arg1, arg2, arg3)`, and
    `%doclink(arg1,arg2)` will be replaced with `link_to_doc_page(arg1, arg2)`.

    USAGE EXAMPLES
    --------------

    %doclink(api,index_pipeline) -> ../apiref/index_pipeline.rst

    %doclink(api,script.core.script) -> ../apiref/chatsky.script.core.script.rst

    %doclink(api,script.core.script,Node) -> ../apiref/chatsky.script.core.script.rst#chatsky.script.core.script.Node

    %doclink(tutorial,messengers.web_api_interface.4_streamlit_chat)) ->
    ../tutorials/tutorials.messengers.web_api_interface.4_streamlit_chat.py

    %doclink(tutorial,messengers.web_api_interface.4_streamlit_chat,API-configuration) ->
    ../tutorials/tutorials.messengers.web_api_interface.4_streamlit_chat.py#API-configuration

    %doclink(guide,basic_conceptions) -> ../user_guides/basic_conceptions.rst

    %doclink(guide,basic_conceptions,example-conversational-chat-bot) ->
    ../user_guides/basic_conceptions.rst#example-conversational-chat-bot

    """

    pattern: ClassVar[re.Pattern] = re.compile(r"%doclink\((.+?)\)")

    @staticmethod
    def link_to_doc_page(
        page_type: Literal["api", "tutorial", "rst"],
        dotpath: str,
        obj: str | None = None,
    ) -> str:
        """
        Create a link to a documentation page.

        :param page_type:
            Type of the documentation:

                - "api" -- API reference
                - "tutorial" -- Tutorials
                - "rst" -- User guides

        :param dotpath:
            Path to the index page in unix style.

            So, to link Dataset, pass "context" as page (omitting the "autointent" prefix).

            To link to the basic script tutorial, pass "script.core.1_basics" (without the "tutorials" prefix).

            To link to the basic concepts guide, pass "basic_conceptions".

            API index pages are also supported.
            Passing "index_pipeline" will link to the "apiref/index_pipeline.html" page.
        :param obj:
            An anchor on the page. (optional)

            For the "api" type, use only the last part of the linked object.

            So, to link to the `CLIMessengerInterface` class, pass "CLIMessengerInterface" only.

            To link to a specific section of a tutorial or a guide, pass an anchor as-is (e.g. "introduction").
        :return:
            A link to the corresponding documentation part.
        """
        if page_type == "class":
            dotpath = "autointent" + (("." + dotpath) if dotpath != "" else "")
            path = "/".join(dotpath.split("."))
            return f"../autoapi/{path}/{obj}.html" + (f"#{dotpath}.{obj}" if obj is not None else "")
        if page_type == "method":
            dotpath = "autointent" + (("." + dotpath) if dotpath != "" else "")
            path = "/".join(dotpath.split("."))
            return f"../autoapi/{path}.html" + (f"#{dotpath}.{obj}" if obj is not None else "")
        if page_type == "tutorial":
            return f"../tutorials/tutorials.{dotpath}.py"
        if page_type == "rst":
            path = "/".join(dotpath.split("."))
            return f"../{path}.rst"
        msg = "Unexpected page type"
        raise ValueError(msg)

    @staticmethod
    def replacement_string(matchobj: re.Match) -> str:
        args = matchobj.group(1).split(",")
        return DocumentationLink.link_to_doc_page(*args)


class MarkdownDocumentationLink(DocumentationLink):
    """
    Replace documentation linking directives with markdown-style links.

    Replace strings of the `%mddoclink({args})` format with corresponding links to local files.

    `args` is a comma-separated string of arguments to pass to the :py:meth:`.DocumentationLink.link_to_doc_page`.

    So, `%mddoclink(arg1,arg2,arg3)` will be replaced with `[text](link_to_doc_page(arg1, arg2, arg3))`, and
    `%doclink(arg1,arg2)` will be replaced with `[text](link_to_doc_page(arg1, arg2))` with `text` being the last
    path segment of the last argument.

    USAGE EXAMPLES
    --------------

    %mddoclink(api,index_pipeline) -> [index_pipeline](
        ../apiref/index_pipeline.rst
    )

    %mddoclink(api,script.core.script,Node) -> [Node](
        ../apiref/chatsky.script.core.script.rst#chatsky.script.core.script.Node
    )

    %mddoclink(tutorial,messengers.web_api_interface.4_streamlit_chat) -> [4_streamlit_chat](
        ../tutorials/tutorials.messengers.web_api_interface.4_streamlit_chat.py
    )

    %mddoclink(tutorial,messengers.web_api_interface.4_streamlit_chat,API-configuration) -> [API-configuration](
        ../tutorials/tutorials.messengers.web_api_interface.4_streamlit_chat.py#API-configuration
    )

    %mddoclink(guide,basic_conceptions) -> [basic_conceptions](
        ../user_guides/basic_conceptions.rst
    )

    %mddoclink(guide,basic_conceptions,example-conversational-chat-bot) -> [example-conversational-chat-bot](
        ../user_guides/basic_conceptions.rst#example-conversational-chat-bot
    )

    """

    pattern: ClassVar[re.Pattern] = re.compile(r"%mddoclink\((.+?)\)")

    @staticmethod
    def replacement_string(matchobj: re.Match) -> str:
        args = matchobj.group(1).split(",")
        link_text = args[-1].split(".")[-1]
        return f"[{link_text}]({DocumentationLink.link_to_doc_page(*args)})"


def apply_replace_patterns(text: str) -> str:
    for cls in (InstallationCell, DocumentationLink, MarkdownDocumentationLink):
        text = cls.replace(text)

    return text


def py_percent_to_notebook(text: str) -> nbformat.NotebookNode:
    """
    Convert `.py`-file to jupyter notebook.

    This function takes string in `py:percent` format, applies replacement patterns
    :py:class:`docs_utils.InstallationCell`, :py:function:`docs_utils.DocumentationLink`,
    :py:function:`docs_utils.MarkdownDocumentationLink` and converts result to standard
    JSON representation of jupyter notebook. See example:

    .. code-block:: python

        # %% [markdown]
        # # Example Notebook
        # This is an example notebook using the `py:percent` format.

        # %%
        # This is a code cell
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        plt.plot(x, y)
        plt.show()

        # %% [markdown]
        # ## Another Markdown Cell
        # This is another markdown cell with some additional text.

        # %% [raw]
        # This is a raw cell
    """
    return jupytext.reads(apply_replace_patterns(text), "py:percent")
