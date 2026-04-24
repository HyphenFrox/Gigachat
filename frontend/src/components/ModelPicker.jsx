import React from 'react'
import { ChevronDown, Wrench } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

/**
 * ModelPicker — compact dropdown that selects the conversation's model.
 *
 * Extracted from ChatHeader so it can live next to the composer (where
 * the "what model will send the next message?" question actually
 * matters). Changing the picker persists via onPatch({ model }); the
 * new model takes effect on the *next* turn — any in-flight stream
 * continues on whichever model started it.
 *
 * Props:
 *   conv                  - current conversation (reads conv.model).
 *   models                - list of model names to show in the menu.
 *   showAllModels         - toggle state for the "show all / tool-only"
 *                           filter footer.
 *   onToggleShowAllModels - called when the user flips the filter.
 *                           Optional — if omitted, the filter footer
 *                           is hidden entirely.
 *   onPatch               - called with ({ model }) when the user picks
 *                           a different model.
 */
export default function ModelPicker({
  conv,
  models,
  showAllModels,
  onToggleShowAllModels,
  onPatch,
}) {
  if (!conv) return null
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1 px-2 text-xs"
          title={conv.model}
        >
          <span className="max-w-[160px] truncate sm:max-w-[240px]">
            {conv.model}
          </span>
          <ChevronDown className="size-3 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      {/* Same width discipline as before: min keeps short names readable,
          max caps it on desktop, break-all lets long slash/colon names
          wrap since they have no natural break points. */}
      <DropdownMenuContent
        align="start"
        className="min-w-[16rem] max-w-[min(90vw,28rem)]"
      >
        <DropdownMenuLabel className="flex items-center justify-between gap-2 text-xs">
          <span>
            {showAllModels ? 'All installed models' : 'Tool-capable models'}
          </span>
          <Wrench
            className={`size-3 ${showAllModels ? 'text-muted-foreground/50' : 'text-emerald-400'}`}
          />
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        {models.length === 0 && (
          <DropdownMenuItem disabled>
            {showAllModels ? 'No models found' : 'No tool-capable models found'}
          </DropdownMenuItem>
        )}
        {models.map((m) => (
          <DropdownMenuItem
            key={m}
            onClick={() => m !== conv.model && onPatch({ model: m })}
            title={m}
            className="items-start gap-2"
          >
            <span className="flex-1 break-all text-xs leading-snug">{m}</span>
            {m === conv.model && (
              <span className="shrink-0 text-[10px] text-muted-foreground">
                current
              </span>
            )}
          </DropdownMenuItem>
        ))}
        {onToggleShowAllModels && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={(e) => {
                e.preventDefault()
                onToggleShowAllModels()
              }}
              className="gap-2 text-xs"
              title={
                showAllModels
                  ? 'Hide models whose Ollama template does not declare tool support (the agent will 400 on them).'
                  : 'Show every installed model, including ones without tool support. Useful if you know a specific model works for your use case.'
              }
            >
              <Wrench className="size-3.5 shrink-0" />
              <span className="flex-1">
                {showAllModels ? 'Only tool-capable' : 'Show all models'}
              </span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
