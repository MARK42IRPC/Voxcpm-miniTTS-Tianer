from . import cleaned_text_to_sequence
import copy


def _language_module(language):
    # Import only the selected frontend. The upstream eager imports make a
    # Chinese-only trainer require every Japanese/Korean/European dependency.
    if language == "ZH":
        from . import chinese

        return chinese
    if language == "ZH_MIX_EN":
        from . import chinese_mix

        return chinese_mix
    raise ValueError(f"This embedded MeloTTS trainer supports ZH and ZH_MIX_EN, got {language}")


def clean_text(text, language):
    language_module = _language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = _language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
