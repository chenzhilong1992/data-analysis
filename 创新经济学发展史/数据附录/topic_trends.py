"""Count economics papers per year that mention each stage's signature topics.

Data source: OpenAlex (https://openalex.org), free and without an API key.
For each keyword group, the script asks OpenAlex for the number of works per
publication year whose title or abstract matches the group, restricted to the
field "Economics, Econometrics and Finance". It divides by the total number of
works in that field for the same year, then writes a CSV and a line chart.

Usage:
    pip install requests pandas matplotlib
    python topic_trends.py --email you@example.com [--start 1950 --end 2024]

The e-mail is sent to OpenAlex as the `mailto` parameter so that requests go to
its "polite pool"; it is not used for anything else.

Caveat: keyword matches only approximate research topics, and coverage of
older abstracts in OpenAlex is uneven, so compare relative trends rather than
absolute levels.
"""
import argparse
import time
from pathlib import Path

import requests

API = 'https://api.openalex.org/works'
# OpenAlex topic taxonomy: field 20 = "Economics, Econometrics and Finance".
FIELD_FILTER = 'primary_topic.field.id:20'
# Fallback for older API behaviour: the legacy "Economics" concept.
CONCEPT_FILTER = 'concepts.id:C162324750'

GROUPS = {
    '第一阶段 创造性破坏/企业家': ['creative destruction', 'entrepreneurship'],
    '第二阶段 增长核算/余值': ['growth accounting', 'Solow residual', 'total factor productivity'],
    '第三阶段 演化/创新系统': ['evolutionary economics', 'national innovation system', 'path dependence'],
    '第四阶段 内生增长': ['endogenous growth', 'increasing returns', 'R&D-based growth'],
    '第五阶段 专利/溢出/风险资本': ['patent citations', 'knowledge spillovers', 'venture capital'],
    '第六阶段 活力/集中/自动化/AI': ['business dynamism', 'markups', 'automation',
                            'artificial intelligence', 'industrial policy'],
}


def search_expr(terms):
    # Boolean OR of exact phrases; commas would break OpenAlex's filter syntax.
    for t in terms:
        assert ',' not in t, t
    return '(' + ' OR '.join(f'"{t}"' for t in terms) + ')'


def counts_by_year(filters, email, start, end):
    params = {
        'filter': ','.join(filters + [f'publication_year:{start}-{end}']),
        'group_by': 'publication_year',
        'per_page': 200,
        'mailto': email,
    }
    for attempt in range(4):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return {int(g['key']): g['count'] for g in r.json()['group_by']}
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:300]}')
    raise RuntimeError('OpenAlex did not respond after retries')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--email', required=True, help='contact e-mail for the OpenAlex polite pool')
    ap.add_argument('--start', type=int, default=1950)
    ap.add_argument('--end', type=int, default=2024)
    ap.add_argument('--out', default=str(Path(__file__).with_name('topic_trends')))
    args = ap.parse_args()

    import pandas as pd
    import matplotlib.pyplot as plt

    field = FIELD_FILTER
    try:
        total = counts_by_year([field], args.email, args.start, args.end)
    except RuntimeError:
        field = CONCEPT_FILTER
        total = counts_by_year([field], args.email, args.start, args.end)

    years = list(range(args.start, args.end + 1))
    df = pd.DataFrame({'year': years, 'economics_total': [total.get(y, 0) for y in years]})
    for name, terms in GROUPS.items():
        c = counts_by_year([field, f'title_and_abstract.search:{search_expr(terms)}'],
                           args.email, args.start, args.end)
        df[name] = [c.get(y, 0) for y in years]
        time.sleep(0.2)
    shares = df[list(GROUPS)].div(df['economics_total'].where(df['economics_total'] > 0), axis=0) * 1000
    shares.insert(0, 'year', df['year'])

    df.to_csv(args.out + '_counts.csv', index=False, encoding='utf-8-sig')
    shares.to_csv(args.out + '_per_1000.csv', index=False, encoding='utf-8-sig')

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    smooth = shares.set_index('year').rolling(3, center=True, min_periods=1).mean()
    for name in GROUPS:
        ax.plot(smooth.index, smooth[name], label=name, linewidth=1.8)
    for x, label in [(1945, '二'), (1970, '三'), (1986, '四'), (1990, '五'), (2008, '六')]:
        ax.axvline(x, color='grey', linewidth=0.6, linestyle=':')
        ax.text(x, ax.get_ylim()[1] * 0.97, f'第{label}阶段', fontsize=8, color='grey', ha='left')
    ax.set_xlabel('年份')
    ax.set_ylabel('每千篇经济学文献中的篇数（三年移动平均）')
    ax.set_title('创新经济学各阶段标志性主题的文献占比')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=200)
    print('wrote', args.out + '_counts.csv', args.out + '_per_1000.csv', args.out + '.png')


if __name__ == '__main__':
    main()
