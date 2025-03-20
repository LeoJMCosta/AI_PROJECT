import re


def clean_text(text):
    # 1. Remove símbolos indesejados (ex.: “”, “”)
    text = re.sub(r'[]', '', text)

    # 2. Substitui quebras de linha simples por espaço (preservando duplas para separar parágrafos)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 3. Junta palavras com letras separadas por espaços (ex.: "V er s ão" → "Versão")
    pattern = r'\b(?:[A-Za-zÀ-ÖØ-öø-ÿ]\s){1,}[A-Za-zÀ-ÖØ-öø-ÿ]\b'
    text = re.sub(pattern, lambda m: m.group(0).replace(' ', ''), text)

    # 4. Remove referências de página (ex.: "Pág. 19 de 245")
    text = re.sub(r'P[áa]g\.\s*\d+\s*de\s*\d+', '', text, flags=re.IGNORECASE)

    # 5. Remove linhas com informações de versão e rodapé
    # Remove qualquer linha que contenha "Versão à data de" seguido de "Pág." (agora unificado)
    text = re.sub(r'^.*Versão à data de.*Pág\..*$', '',
                  text, flags=re.IGNORECASE | re.MULTILINE)

    # 6. Remove cabeçalhos/rodapés fixos conhecidos
    text = re.sub(r'CÓDIGO DO TRABALHO\s*-\s*CT',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'LEGISLA[CÇ]A CONSOLIDADA', '', text, flags=re.IGNORECASE)

    # 7. Normaliza espaços em branco
    text = re.sub(r'\s+', ' ', text).strip()

    return text
