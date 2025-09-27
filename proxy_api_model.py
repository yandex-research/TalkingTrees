"""
Implements ProxyAPIModel that calls a model over Openai-like proxy API for use in smolagents
"""
from typing import Optional
import smolagents
import requests
from smolagents import ChatMessage, TokenUsage
import openai


class ProxyAPIModel(smolagents.OpenAIServerModel):
    def __init__(self, *args, max_new_tokens: int, callback: Optional[callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.callback = callback

    def generate_stream(self, *args, **kwargs):
        raise NotImplementedError('TODO')

    def generate(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from=None,
            **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = requests.post(
            self.client_kwargs['base_url'],
            headers={"authorization": f"OAuth {self.client_kwargs['api_key']}",
                     "content-type": "application/json"},
            verify=False,
            json=dict(
                **completion_kwargs,
                # max_completion_tokens=self.max_new_tokens
            )
        ).json()
        response = openai.types.chat.chat_completion.ChatCompletion(
            id=response['response']['id'], model=response['response']['model'], created=response['response']['created'],
            choices=[openai.types.chat.chat_completion.Choice(**c) for c in response['response']['choices']],
            usage=openai.types.CompletionUsage(
                prompt_tokens=response['usage']['input_tokens'],
                completion_tokens=response['usage']['output_tokens'],
                total_tokens=response['usage']['total_tokens']
            ),
            object='chat.completion'
        )

        # TODO this does not parse all fields in response.response; there might be a better way

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        response_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
        if self.callback is not None:
            self.callback(response_message, completion_kwargs=completion_kwargs)
        return response_message
