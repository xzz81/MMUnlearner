import torch


def train_collate_fn(examples, processor, device, train_flag=True):
    """标准 collate 函数，计算完整序列的 loss"""
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')

        if image is None:
            user_content = [{"type": "text", "text": question}]
        else:
            user_content = [{"type": "image"}, {"type": "text", "text": question}]
            images.append(image)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())

    if len(texts) == 0:
        raise ValueError("Empty batch")

    batch = processor(
        text=texts,
        images=images if len(images) > 0 else None,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if train_flag and device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def train_collate_fn_ansonly(examples, processor, device, train_flag=True):
    """只计算 answer 部分 loss 的 collate 函数"""
    images = []
    texts = []
    answer_ids = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')

        if image is None:
            user_content = [{"type": "text", "text": question}]
        else:
            user_content = [{"type": "image"}, {"type": "text", "text": question}]
            images.append(image)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())

        # 提取 answer token ids
        answer_token = processor.tokenizer(answer, return_tensors="pt")
        all_special_ids = torch.Tensor(processor.tokenizer.all_special_ids)
        answer_mask = torch.isin(answer_token['input_ids'][0], all_special_ids, invert=True)
        answer_token = answer_token['input_ids'][0][answer_mask]
        answer_ids.append(answer_token)

    if len(texts) == 0:
        raise ValueError("Empty batch")

    batch = processor(
        text=texts,
        images=images if len(images) > 0 else None,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 只保留 answer 部分的 labels
    labels = batch["input_ids"].clone()
    for label, answer_id in zip(labels, answer_ids):
        found = False
        for idx in range(len(label) - len(answer_id) + 1):
            if torch.equal(label[idx: idx + len(answer_id)], answer_id):
                found = True
                label[:idx] = -100
                label[idx + len(answer_id):] = -100
                break
        if not found:
            label[:] = -100

    batch["labels"] = labels

    if train_flag and device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
