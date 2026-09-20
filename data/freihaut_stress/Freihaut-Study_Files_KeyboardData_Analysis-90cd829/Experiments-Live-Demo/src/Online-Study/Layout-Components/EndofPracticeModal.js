// React Imports
import React from 'react';

// end of task modal
export default function EndOfPractice (props) {

    return(
        <div className={props.startModal} style={{textAlign: "left", fontSize: "18px"}}>
            <div className="modal-background">{null}</div>
            <div className="modal-content">
                <header className="modal-card-head">
                    <p className="modal-card-title">{null}</p>
                </header>
                <section className="modal-card-body">
                    <div className="content">
                        <h3>Task clear?</h3><br/>
                        <p>
                            If you are still unclear how to work on the task, you can reread the task instruction by clicking on „Restart practice task“.
                        </p>
                        <p>
                            If it is clear to you how to work on the task, click "continue to work on the task“.
                        </p>
                    </div>
                </section>
                <footer className="modal-card-foot">
                    <div className={"field is-grouped"}>
                        <button className="button is-link" onClick={() => props.restart()}>Restart practice task</button>
                        <button className="button is-link" onClick={() => props.initializeTask()}>Continue to work on the task</button>
                    </div>
                </footer>
            </div>
        </div>

    )
}