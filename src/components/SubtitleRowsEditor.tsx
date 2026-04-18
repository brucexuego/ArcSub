import React from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { EditableSubtitleRow, SubtitleEditorIssue, makeSubtitleRow } from '../utils/subtitle_editor';

interface SubtitleRowsEditorCopy {
  timecodeLabel: string;
  textLabel: string;
  addRow: string;
  deleteRow: string;
  timecodePlaceholder: string;
  textPlaceholder: string;
  emptyHint: string;
  timecodeFormatError: string;
  timecodeBeforePreviousError: string;
  timecodeAfterNextError: string;
  textRequiredError: string;
}

interface SubtitleRowsEditorProps {
  rows: EditableSubtitleRow[];
  issues: SubtitleEditorIssue[];
  onChangeRows: (rows: EditableSubtitleRow[]) => void;
  copy: SubtitleRowsEditorCopy;
  referenceLines?: string[];
}

function issueCodeToMessage(issue: SubtitleEditorIssue, copy: SubtitleRowsEditorCopy) {
  switch (issue.code) {
    case 'timecode_format':
      return copy.timecodeFormatError;
    case 'timecode_before_previous':
      return copy.timecodeBeforePreviousError;
    case 'timecode_after_next':
      return copy.timecodeAfterNextError;
    case 'text_required':
      return copy.textRequiredError;
    default:
      return '';
  }
}

function combineIssueMessages(issueList: SubtitleEditorIssue[], copy: SubtitleRowsEditorCopy) {
  const seen = new Set<string>();
  return issueList
    .map((issue) => issueCodeToMessage(issue, copy))
    .filter(Boolean)
    .filter((message) => {
      if (seen.has(message)) return false;
      seen.add(message);
      return true;
    });
}

export default function SubtitleRowsEditor({ rows, issues, onChangeRows, copy, referenceLines }: SubtitleRowsEditorProps) {
  const issuesByIndex = React.useMemo(() => {
    const map = new Map<number, SubtitleEditorIssue[]>();
    for (const issue of issues) {
      const current = map.get(issue.index) || [];
      current.push(issue);
      map.set(issue.index, current);
    }
    return map;
  }, [issues]);

  const handleChangeTimecode = React.useCallback(
    (index: number, value: string) => {
      const nextRows = rows.map((row, rowIndex) => (
        rowIndex === index
          ? { ...row, timecode: value }
          : row
      ));
      onChangeRows(nextRows);
    },
    [onChangeRows, rows]
  );

  const handleChangeText = React.useCallback(
    (index: number, value: string) => {
      const nextRows = rows.map((row, rowIndex) => (
        rowIndex === index
          ? { ...row, text: value }
          : row
      ));
      onChangeRows(nextRows);
    },
    [onChangeRows, rows]
  );

  const handleInsertAfter = React.useCallback(
    (index: number) => {
      const nextRows = [...rows];
      nextRows.splice(index + 1, 0, makeSubtitleRow('', ''));
      onChangeRows(nextRows);
    },
    [onChangeRows, rows]
  );

  const handleDelete = React.useCallback(
    (index: number) => {
      const nextRows = rows.filter((_, rowIndex) => rowIndex !== index);
      onChangeRows(nextRows);
    },
    [onChangeRows, rows]
  );

  const handleAppend = React.useCallback(() => {
    onChangeRows([...rows, makeSubtitleRow('', '')]);
  }, [onChangeRows, rows]);

  const hasReference = Array.isArray(referenceLines);

  return (
    <div className="space-y-3">
      {rows.length === 0 && (
        <div className="rounded-xl border border-dashed border-white/10 bg-surface-container-lowest/50 px-4 py-5 text-xs text-outline/80">
          {copy.emptyHint}
        </div>
      )}

      <div className="space-y-3">
        {rows.map((row, index) => {
          const rowIssues = issuesByIndex.get(index) || [];
          const rowIssueMessages = combineIssueMessages(rowIssues, copy);
          const referenceLine = hasReference ? String(referenceLines?.[index] || '').trim() : '';
          const editorNode = (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-44 shrink-0">
                  <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-outline/75">{copy.timecodeLabel}</label>
                  <input
                    value={row.timecode}
                    onChange={(e) => handleChangeTimecode(index, e.target.value)}
                    placeholder={copy.timecodePlaceholder}
                    className={`w-full rounded-lg border bg-surface-container-lowest px-3 py-2 font-mono text-[12px] text-secondary outline-none focus:ring-2 focus:ring-primary-container ${
                      rowIssues.some((issue) => issue.field === 'timecode')
                        ? 'border-error/50'
                        : 'border-white/10'
                    }`}
                  />
                </div>
                <div className="ml-auto flex items-center gap-1.5 pt-4">
                  <button
                    type="button"
                    onClick={() => handleInsertAfter(index)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-outline transition-colors hover:bg-white/10 hover:text-secondary"
                    title={copy.addRow}
                  >
                    <Plus className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(index)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-outline transition-colors hover:bg-error/20 hover:text-error"
                    title={copy.deleteRow}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <div>
                <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-outline/75">{copy.textLabel}</label>
                <textarea
                  value={row.text}
                  onChange={(e) => handleChangeText(index, e.target.value)}
                  rows={2}
                  placeholder={copy.textPlaceholder}
                  className={`w-full resize-y rounded-xl border bg-surface-container-lowest px-3 py-2 text-sm leading-relaxed text-secondary outline-none focus:ring-2 focus:ring-primary-container ${
                    rowIssues.some((issue) => issue.field === 'text')
                      ? 'border-error/50'
                      : 'border-white/10'
                  }`}
                />
              </div>
              {rowIssueMessages.length > 0 && (
                <div className="space-y-1 text-[11px] text-error">
                  {rowIssueMessages.map((message) => (
                    <div key={`${row.id}-${message}`}>{message}</div>
                  ))}
                </div>
              )}
            </div>
          );

          if (!hasReference) {
            return (
              <div key={row.id} className="rounded-xl border border-white/8 bg-surface-container-high/40 p-3">
                {editorNode}
              </div>
            );
          }

          return (
            <div key={row.id} className="grid grid-cols-2 overflow-hidden rounded-xl border border-white/8">
              <div className="border-r border-white/5 bg-surface-container-lowest/55 px-4 py-4 text-sm leading-relaxed text-outline/85 whitespace-pre-wrap">
                {referenceLine || '-'}
              </div>
              <div className="bg-primary/5 p-3">
                {editorNode}
              </div>
            </div>
          );
        })}
      </div>

      <button
        type="button"
        onClick={handleAppend}
        className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs font-bold uppercase tracking-widest text-outline transition-colors hover:bg-white/10 hover:text-secondary"
      >
        <Plus className="h-3.5 w-3.5" />
        {copy.addRow}
      </button>
    </div>
  );
}
