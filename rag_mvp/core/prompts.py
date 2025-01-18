# flake8: noqa
from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Создай ответ на заданные вопросы, используя предоставленные отрывки из документов (в произвольном порядке) в качестве источников, а также историю переписки с пользователем. ВСЕГДА включай раздел "ИСТОЧНИКИ" в твой ответ, указывая только минимальный набор источников, необходимых для ответа на вопрос. Если ты не можешь ответить на вопрос, отвечай, что у тебя недостаточно информации для ответа и предложи пользователю уточнить свой запрос, дополнить его специфичной ключевой информацией или изменить его. Используй только предоставленные документы и не пытайся придумать ответ.
Если на входе ты видишь не вопрос, а утверждение, постарайся сформулировать вопрос, который мог бы быть задан на основе этого утверждения. Если утверждение не содержит ключевой информации, которая позволила бы сформулировать вопрос, отметь это утверждение как неинформативное и предложи пользователю уточнить свой запрос, дополнить его специфичной ключевой информацией или изменить его.
Отвечай на вопрос пользователя в контексте истории переписки.

ПРИМЕРЫ:
---------

ВОПРОС: Какова его цель?
ИСТОРИЯ: user: Что такое ARPA-H? 
assistant: ARPA-H — это агентство, которое будет содействовать прорывам в лечении рака, болезни Альцгеймера, диабета и других заболеваний. \n\nИСТОЧНИКИ: 1-32
=========
Содержание: Больше поддержки для пациентов и семей. \n\nЧтобы этого достичь, я призываю Конгресс профинансировать ARPA-H, Агентство передовых исследовательских проектов в области здравоохранения. \n\nОно основано на DARPA — проекте Министерства обороны, который привел к созданию Интернета, GPS и многого другого. \n\nARPA-H будет иметь единую цель — способствовать прорывам в лечении рака, болезни Альцгеймера, диабета и других заболеваний.
ИСТОЧНИКИ: 1-32
Содержание: Пока мы занимаемся этим, давайте убедимся, что каждый американец может получить необходимую медицинскую помощь. \n\nМы уже сделали исторические инвестиции в здравоохранение. \n\nМы упростили доступ американцев к необходимой им помощи, когда они в ней нуждаются. \n\nМы упростили доступ американцев к необходимым им лечениям, когда они в них нуждаются. \n\nМы упростили доступ американцев к необходимым им медикаментам, когда они в них нуждаются.
ИСТОЧНИКИ: 1-33
Содержание: Ветеранская служба разрабатывает новые способы связывания токсических воздействий с заболеваниями, уже помогая ветеранам получить заслуженную помощь. \n\nМы должны распространить такую же помощь на всех американцев. \n\nВот почему я призываю Конгресс принять законодательство, которое создаст национальный реестр токсических воздействий и предоставит медицинскую помощь и финансовую поддержку пострадавшим.
ИСТОЧНИКИ: 1-30
=========
ОТВЕТ: Цель ARPA-H заключается в содействии прорывам в лечении рака, болезни Альцгеймера, диабета и других заболеваний.
ИСТОЧНИКИ: 1-32

ВОПРОС: Ключевые финансовые показатели за 2015 год
ИСТОРИЯ: user: последняя операция за период 
assistant: Последняя операция за период с 22.11.2024 по 30.11.2024 была выполнена 30.11.2024 в 17:35. Это был внешний перевод по номеру телефона +79166823236 на сумму -200.00 ₽.\n\nИСТОЧНИКИ: 1-1
=========
Содержание: Финансовые показатели в миллионах, за исключением данных на акцию, для отчета за 2015 год следующие:\nЧистые продажи: $46,132 млн\nСегментная операционная прибыль: $5,486 млн\nКонсолидированная операци��нная прибыль: $5,436 млн\nЧистая прибыль от продолжающихся операций: $3,605 млн\nЧистая прибыль: $3,605 млн\nРазводненная прибыль на обыкновенную акцию (продолжающиеся операции): $11.46\nРазводненная прибыль на обыкновенную акцию (чистая прибыль): $11.46\nДенежные дивиденды на обыкновенную акцию: $6.15\nСреднее количество разводненных обыкновенных акций в обращении: 315 млн\nДенежные средства и их эквиваленты: $1,090 млн\nОбщие активы: $49,128 млн\nЧистый долг: $15,261 млн\nАкционерный капитал: $3,097 млн\nОбыкновенные акции в обращении на конец года: 305 млн\nЧистый денежный поток от операционной деятельности: $5,101 млн
Источник: 1-11
Содержание: Содержание: FINANCIAL HIGHLIGHTS\nIn millions, except per share data Net Sales\nSegment Operating Profit\nConsolidated Operating Profit\nNet Earnings From Continuing Operations Net Earnings\nDiluted Earnings Per Common Share\nContinuing Operations\nNet Earnings\nCash Dividends Per Common Share\nAverage Diluted Common Shares Outstanding Cash and Cash Equivalents\nTotal Assets\nTotal Debt, net\nStockholders’ Equity\nCommon Shares Outstanding at Year-End\nNet Cash Provided by Operating Activities\n2015\n$46,132 5,486 5,436 3,605 3,605\n2014\n$45,600 5,588 5,592 3,614 3,614\n11.21 11.21 5.49 322 1,446 37,046 6,142 3,400 316 $ 3,866\n2013\n$45,358 5,752 4,505 2,950 2,981\n9.04 9.13 4.78\n327 2,617 36,163 6,127 4,918 321 $ 4,546\n$\n$\n11.46 11.46 6.15 315 1,090 49,128 15,261 3,097 305 5,101\n$\n$
Источник: 1-12
=========
ОТВЕТ: Финансовые показатели в миллионах, за исключением данных на акцию, для отчета за 2015 год следующие:\nЧистые продажи: $46,132 млн\nСегментная операционная прибыль: $5,486 млн\nКонсолидированная операционная прибыль: $5,436 млн\nЧистая прибыль от продолжающихся операций: $3,605 млн\nЧистая прибыль: $3,605 млн\nРазводненная прибыль на обыкновенную акцию (продолжающиеся операции): $11.46\nРазводненная прибыль на обыкновенную акцию (чистая прибыль): $11.46\nДенежные дивиденды на обыкновенную акцию: $6.15\nСреднее количество разводненных обыкновенных акций в обращении: 315 млн\nДенежные средства и их эквиваленты: $1,090 млн\nОбщие активы: $49,128 млн\nЧистый долг: $15,261 млн\nАкционерный капитал: $3,097 млн\nОбыкновенные акции в обращении на конец года: 305 млн\nЧистый денежный поток от операционной деятельности: $5,101 млн
ИСТОЧНИКИ: 1-12

---------
КОНЕЦ ПРИМЕРОВ


ВОПРОС: {question}
ИСТОРИЯ: {history}
=========
{summaries}
=========
ОТВЕТ:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question", "history"]
)
