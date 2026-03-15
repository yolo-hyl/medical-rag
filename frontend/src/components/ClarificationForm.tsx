import { useState } from 'react';
import { X, Send } from 'lucide-react';

interface ClarificationFormProps {
  questions: string[];
  onSubmit: (combined: string) => void;
  onClose: () => void;
}

export default function ClarificationForm({ questions, onSubmit, onClose }: ClarificationFormProps) {
  const [answers, setAnswers] = useState<string[]>(() => questions.map(() => ''));

  const handleSubmit = () => {
    const combined = questions
      .map((q, i) => `${q}：${answers[i].trim()}`)
      .filter((_, i) => answers[i].trim())
      .join('；');
    if (!combined) return;
    onSubmit(combined);
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/10 z-10"
        onClick={onClose}
      />
      {/* Slide-up drawer */}
      <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 shadow-xl rounded-t-2xl z-20 px-5 py-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-800">请补充以下信息</h3>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X size={15} />
          </button>
        </div>

        <div className="space-y-3 max-h-60 overflow-y-auto scrollbar-thin">
          {questions.map((q, i) => (
            <div key={i}>
              <label className="block text-xs font-medium text-slate-600 mb-1">{q}</label>
              <input
                type="text"
                value={answers[i]}
                onChange={(e) => {
                  const next = [...answers];
                  next[i] = e.target.value;
                  setAnswers(next);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSubmit();
                }}
                placeholder="请输入..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent"
              />
            </div>
          ))}
        </div>

        <div className="flex justify-end mt-4">
          <button
            onClick={handleSubmit}
            disabled={!answers.some((a) => a.trim())}
            className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Send size={14} />
            提交回答
          </button>
        </div>
      </div>
    </>
  );
}
