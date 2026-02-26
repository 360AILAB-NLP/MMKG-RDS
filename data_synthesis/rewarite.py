from llms.client import *
import asyncio

class ReWriter:  
    def __init__(
        self,
        client: AsyncLLMClient,
    ):
        pass

    def make_message(sys_prompt, user_prompt):
        message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        return message

    async def rewrite(qa_info, sys_prompt, user_prompt):
        pass

    async def rewrite_batch(qa_info_list, sys_prompt, user_prompt):
        messages = []
        for qa_info in qa_info_list:
            message = self.make_message(sys_prompt, user_prompt.format(**qa_info))
            messages.append(message)
        
        responses = await self.client.agenerate_batch(messages)
        for qa_info, response in zip(qa_info_list, responses):
            qa_info["rewrite_info"] = response['choices'][0]['message']['content'].strip()
        return qa_info_list